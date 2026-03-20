"""
IBEX 35 — Backend FastAPI
Endpoints: /health, /predict/5days, /historical, /metrics, /accuracy
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
STATIC_DIR = Path(__file__).parent / "static"

SEQUENCE_LENGTH = 60
N_FEATURES      = 1   # Close (univariado)

# ── Startup: cargar modelos ────────────────────────────────────────────────
print("Cargando modelos...")
try:
    from tensorflow import keras
    lstm_model = keras.models.load_model(MODELS_DIR / "lstm_model.keras")
    print("LSTM cargado.")
except Exception as e:
    lstm_model = None
    print(f"Advertencia: No se pudo cargar LSTM: {e}")

try:
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    print("Scaler cargado.")
except Exception as e:
    scaler = None
    print(f"Advertencia: No se pudo cargar scaler: {e}")

try:
    metrics_df   = pd.read_csv(DATA_DIR / "metrics_comparison.csv")
    metrics_dict = metrics_df.to_dict(orient="records")
except Exception:
    metrics_dict = []

try:
    from src.prediction_logger import save_prediction, get_accuracy_summary
    _logger_available = True
except Exception:
    _logger_available = False

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="IBEX 35 Predictor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Cache en memoria (evita rate limiting de Yahoo Finance) ────────────────
_cache: dict = {"df": None, "ts": None, "days": None}
CACHE_TTL = 3600  # segundos (1 hora)


# ── Helpers ────────────────────────────────────────────────────────────────
def fetch_recent_ibex(days: int = 90) -> pd.DataFrame:
    """Descarga datos recientes del IBEX 35 con caché de 1 hora."""
    now = datetime.now()

    # Devolver caché si está fresco y cubre los días pedidos
    if (_cache["df"] is not None and _cache["ts"] is not None
            and (_cache["days"] or 0) >= days
            and (now - _cache["ts"]).total_seconds() < CACHE_TTL):
        return _cache["df"]

    end       = now
    start     = end - timedelta(days=days + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    for attempt in range(3):
        try:
            data = yf.download("^IBEX", start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df = data[["Close", "Volume"]].dropna()
            if len(df) > 0:
                _cache["df"]   = df
                _cache["ts"]   = now
                _cache["days"] = days
                return df
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2)

    # Si falló pero hay caché antiguo, usarlo antes de devolver error
    if _cache["df"] is not None:
        return _cache["df"]

    raise HTTPException(
        status_code=503,
        detail="No se pudo obtener datos de Yahoo Finance. Intenta de nuevo en unos segundos."
    )


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame con solo Close (modelo univariado)."""
    return df[["Close"]].dropna()


def _inv_close(scaled_arr: np.ndarray) -> np.ndarray:
    """Inverse-transform valores de Close usando el scaler de 4 features."""
    dummy = np.zeros((len(scaled_arr), N_FEATURES))
    dummy[:, 0] = scaled_arr
    return scaler.inverse_transform(dummy)[:, 0]


def predict_5days(df: pd.DataFrame) -> list:
    """Prediccion autoregresiva a 5 dias con el modelo LSTM multivariate."""
    if lstm_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    features = _compute_features(df)
    if len(features) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=400, detail="Datos insuficientes")

    last_60  = features.values[-SEQUENCE_LENGTH:]          # (60, 3)
    scaled   = scaler.transform(last_60)                   # (60, 3)
    input_seq = scaled.tolist()

    preds_scaled = []
    for _ in range(5):
        X = np.array(input_seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        p = float(lstm_model.predict(X, verbose=0)[0, 0])
        preds_scaled.append(p)
        input_seq.append([p])

    return _inv_close(np.array(preds_scaled)).tolist()


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lstm_loaded":   lstm_model is not None,
        "scaler_loaded": scaler is not None,
        "n_features":    N_FEATURES,
        "timestamp":     datetime.now().isoformat(),
    }


@app.get("/predict/5days")
def predict_next_5_days():
    """Genera prediccion a 5 dias habiles usando datos en tiempo real."""
    try:
        df = fetch_recent_ibex(days=120)
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=6)[1:]
        predictions  = predict_5days(df)

        result = []
        for date, price in zip(future_dates, predictions):
            diff = price - last_price
            result.append({
                "fecha":         date.strftime("%Y-%m-%d"),
                "prediccion":    round(price, 2),
                "variacion":     round(diff, 2),
                "variacion_pct": round(diff / last_price * 100, 2),
            })

        if _logger_available and result:
            try:
                save_prediction(
                    precio_base=last_price,
                    prediccion_d1=result[0]["prediccion"],
                    fecha=last_date.strftime("%Y-%m-%d"),
                )
            except Exception:
                pass

        return {
            "ultimo_precio": round(last_price, 2),
            "ultima_fecha":  last_date.strftime("%Y-%m-%d"),
            "predicciones":  result,
            "disclaimer":    "Prediccion orientativa. No constituye asesoramiento financiero.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical")
def get_historical(days: int = 365):
    """Devuelve datos historicos del IBEX 35."""
    try:
        days  = min(days, 3650)
        df    = fetch_recent_ibex(days=days)
        close = df["Close"]
        return {
            "dates":  [d.strftime("%Y-%m-%d") for d in close.index],
            "prices": [round(float(p), 2) for p in close.values],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Devuelve metricas comparativas ARIMA vs LSTM."""
    return {"modelos": metrics_dict}


@app.get("/accuracy")
def get_accuracy():
    """Devuelve historial de predicciones vs reales y metricas de accuracy."""
    if not _logger_available:
        raise HTTPException(status_code=503, detail="Logger no disponible")
    try:
        return get_accuracy_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
