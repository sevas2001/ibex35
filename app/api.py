"""
IBEX 35 — Backend FastAPI
Endpoints: /health, /predict/5days, /historical, /metrics, /accuracy
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = Path(__file__).parent / "static"

SEQUENCE_LENGTH = 60

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
    metrics_df = pd.read_csv(DATA_DIR / "metrics_comparison.csv")
    metrics_dict = metrics_df.to_dict(orient="records")
except Exception:
    metrics_dict = []

try:
    from src.prediction_logger import save_prediction, get_accuracy_summary
    _logger_available = True
except Exception:
    _logger_available = False

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="IBEX 35 Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────
def fetch_recent_ibex(days: int = 90) -> pd.Series:
    """Descarga datos recientes del IBEX 35 con reintentos."""
    import time
    end = datetime.now()
    start = end - timedelta(days=days + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    for attempt in range(3):
        try:
            data = yf.download("^IBEX", start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            close = data["Close"].dropna()
            if len(close) > 0:
                return close
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2)

    raise HTTPException(status_code=503, detail="No se pudo obtener datos de Yahoo Finance. Intenta de nuevo en unos segundos.")


def predict_5days(close_series: pd.Series) -> list:
    """Predicción autoregresiva a 5 días con el modelo LSTM."""
    if lstm_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    last_60 = close_series.values[-SEQUENCE_LENGTH:]
    scaled = scaler.transform(last_60.reshape(-1, 1))
    input_seq = scaled.tolist()
    predictions = []

    for _ in range(5):
        X = np.array(input_seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        pred_scaled = float(lstm_model.predict(X, verbose=0)[0, 0])
        predictions.append(pred_scaled)
        input_seq.append([pred_scaled])

    predictions_real = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten().tolist()
    return predictions_real


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lstm_loaded": lstm_model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/predict/5days")
def predict_next_5_days():
    """Genera predicción a 5 días hábiles usando datos en tiempo real."""
    try:
        close = fetch_recent_ibex(days=120)
        if len(close) < SEQUENCE_LENGTH:
            raise HTTPException(status_code=400, detail="Datos insuficientes")

        last_price = float(close.iloc[-1])
        last_date = close.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=6)[1:]
        predictions = predict_5days(close)

        result = []
        for date, price in zip(future_dates, predictions):
            diff = price - last_price
            result.append({
                "fecha": date.strftime("%Y-%m-%d"),
                "prediccion": round(price, 2),
                "variacion": round(diff, 2),
                "variacion_pct": round(diff / last_price * 100, 2),
            })

        # Guardar predicción del día 1 en el log automáticamente
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
            "ultima_fecha": last_date.strftime("%Y-%m-%d"),
            "predicciones": result,
            "disclaimer": "Predicción orientativa. No constituye asesoramiento financiero.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical")
def get_historical(days: int = 365):
    """Devuelve datos históricos del IBEX 35."""
    try:
        days = min(days, 3650)
        close = fetch_recent_ibex(days=days)
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in close.index],
            "prices": [round(float(p), 2) for p in close.values],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Devuelve métricas comparativas ARIMA vs LSTM."""
    return {"modelos": metrics_dict}


@app.get("/accuracy")
def get_accuracy():
    """Devuelve historial de predicciones vs reales y métricas de accuracy."""
    if not _logger_available:
        raise HTTPException(status_code=503, detail="Logger no disponible")
    try:
        return get_accuracy_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
