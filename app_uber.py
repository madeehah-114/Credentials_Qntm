# app_uber.py
# Full FastAPI app: Quantum stock predictor + auth + robust model loading
import os
from datetime import datetime, timedelta
from typing import Generator

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from fastapi import Depends, FastAPI, Form, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from torch import nn

# ---------------------------
# Configuration
# ---------------------------
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "change_this_to_a_random_secret_in_prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

DATABASE_URL = "sqlite:///./users.db"
SCALER_PATH = "uber_scaler.pkl"
MODEL_PATH = "uber_qnn_model.pt"
CSV_PATH = "UBER.csv"

# Quantum model hyperparams (must match training or adapt loader mapping)
n_qubits = 4
n_layers = 3
n_steps = 10

# ---------------------------
# FastAPI init + templates/static
# ---------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Database (SQLAlchemy sync)
# ---------------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


# initialize DB at import
init_db()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------
# Auth utilities
# ---------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        raise RuntimeError("Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            raise RuntimeError("Invalid token payload")
    except JWTError:
        raise RuntimeError("Could not validate credentials")
    user = get_user_by_email(db, email)
    if user is None:
        raise RuntimeError("User not found")
    return user


# ---------------------------
# Load CSV + scaler
# ---------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found in project root.")
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"{SCALER_PATH} not found. Save MinMaxScaler to this path during training.")
scaler = joblib.load(SCALER_PATH)


# ---------------------------
# Quantum circuit + qlayer
# ---------------------------
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumRegressor(nn.Module):
    def __init__(self, qlayer):
        super().__init__()
        self.qlayer = qlayer
        # classical head used in training scripts
        self.head = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        q_out = self.qlayer(x)
        return self.head(q_out)


# ---------------------------
# Robust model loader
# ---------------------------
def load_model_robust(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found.")

    saved = torch.load(model_path, map_location=torch.device("cpu"))
    canonical = QuantumRegressor(qlayer)
    seq_model = nn.Sequential(qlayer)

    # Try to load full canonical
    try:
        canonical.load_state_dict(saved)
        canonical.eval()
        print("Loaded full QuantumRegressor state_dict.")
        return canonical
    except Exception as e:
        print("Full load failed (this is ok if saved differently):", e)

    # Try to load as sequential (qlayer-only)
    try:
        seq_model.load_state_dict(saved)
        seq_model.eval()
        print("Loaded state into nn.Sequential(qlayer). Using seq_model for inference.")
        return seq_model
    except Exception as e:
        print("Sequential load failed:", e)

    # Try to map '0.*' or 'qlayer.*' to qlayer state dict
    mapped = {}
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k.startswith("0."):
                mapped[k[len("0."):]] = v
            elif k.startswith("qlayer."):
                mapped[k[len("qlayer."):]] = v
            elif k in ("weights",):
                mapped[k] = v

    if mapped:
        try:
            canonical.qlayer.load_state_dict(mapped)
            canonical.eval()
            print("Loaded mapped qlayer weights into canonical model (head random init).")
            return canonical
        except Exception as e:
            print("Mapped qlayer load failed:", e)

    # If here, nothing matched — print sample keys and raise
    print("Saved keys sample (first 50):")
    if isinstance(saved, dict):
        for i, k in enumerate(list(saved.keys())[:50]):
            print(i + 1, k)
    raise RuntimeError("Could not auto-load model. Saved state doesn't match known formats.")


# Load model_for_inference at import time (or raise clear error)
model_for_inference = load_model_robust(MODEL_PATH)


# ---------------------------
# Forecasting helper
# ---------------------------
def forecast_next_days_uber(df_local: pd.DataFrame, n_days: int = 5, n_steps_local: int = n_steps):
    """
    Returns list of {"date": "YYYY-MM-DD", "pred_close": float}
    Uses the global `model_for_inference`.
    """
    if model_for_inference is None:
        raise RuntimeError("Inference model not loaded")

    close_prices = df_local["Close"].values.astype(float)
    forecasted = []
    last_sequence = close_prices[-n_steps_local:].copy()

    for day in range(1, n_days + 1):
        seq_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        if len(seq_scaled) < n_qubits:
            seq_input = np.pad(seq_scaled, (0, n_qubits - len(seq_scaled)), "constant")
        else:
            seq_input = seq_scaled[:n_qubits]

        seq_tensor = torch.tensor(seq_input.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            out = model_for_inference(seq_tensor)
            pred_scaled = out.detach().cpu().numpy().reshape(-1)[0]
        pred_close = float(scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0])

        forecast_date = df_local["Date"].max() + pd.Timedelta(days=day)
        forecasted.append({"date": forecast_date.strftime("%Y-%m-%d"), "pred_close": pred_close})

        # slide the window (real-scale)
        last_sequence = np.append(last_sequence[1:], pred_close)

    return forecasted


# ---------------------------
# Routes: register / login / logout / home / forecast
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    recent_n = 30
    # user detection: try to get current user, but don't fail publicly:
    user = None
    try:
        user = get_current_user(request, db=next(get_db()))
    except Exception:
        user = None

    return templates.TemplateResponse(
        "index_uber.html",
        {
            "request": request,
            "forecast": None,
            "chart_type": "line",
            "actual_dates": df["Date"].dt.strftime("%Y-%m-%d").tolist()[-recent_n:],
            "actual_values": df["Close"].tolist()[-recent_n:],
            "user": user,
        },
    )


@app.get("/register", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
def register_post(request: Request, email: str = Form(...), password: str = Form(...), full_name: str = Form(""), db: Session = Depends(get_db)):
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must be at least 6 characters"})
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"})
    hashed = get_password_hash(password)
    user = User(email=email, hashed_password=hashed, full_name=full_name)
    db.add(user)
    db.commit()
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
def login_post(request: Request, response: Response, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = authenticate_user(db, email, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    token = create_access_token({"sub": user.email})
    resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    resp.set_cookie(key="access_token", value=token, httponly=True, samesite="lax")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie("access_token")
    return resp


@app.post("/forecast", response_class=HTMLResponse)
def forecast(request: Request, days: int = Form(...), chart_type: str = Form(...), user=Depends(get_current_user)):
    # restrict days to reasonable bounds
    days = max(1, min(30, int(days)))
    forecasted = forecast_next_days_uber(df, n_days=days)
    forecast_df = pd.DataFrame(forecasted)
    recent_n = 30
    return templates.TemplateResponse(
        "index_uber.html",
        {
            "request": request,
            "forecast": forecasted,
            "chart_type": chart_type,
            "actual_dates": df["Date"].dt.strftime("%Y-%m-%d").tolist()[-recent_n:],
            "actual_values": df["Close"].tolist()[-recent_n:],
            "pred_dates": forecast_df["date"].tolist(),
            "pred_values": forecast_df["pred_close"].tolist(),
            "user": user,
        },
    )
