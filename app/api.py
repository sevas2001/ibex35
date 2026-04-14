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

# ── Startup: cargar modelos ────────────────────────────────────────────────
def _is_lfs_pointer(path: Path) -> bool:
    """Devuelve True si el archivo es un puntero LFS (no el binario real)."""
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs")
    except Exception:
        return False

def _try_resolve_lfs(path: Path):
    """Intenta resolver punteros LFS via git lfs pull (best-effort)."""
    import subprocess
    try:
        rel = str(path.relative_to(BASE_DIR))
        result = subprocess.run(
            ["git", "lfs", "pull", f"--include={rel}"],
            cwd=str(BASE_DIR), capture_output=True, timeout=30,
        )
        print(f"git lfs pull {rel}: {'OK' if result.returncode == 0 else 'failed'}")
    except Exception as e:
        print(f"No se pudo resolver LFS para {path.name}: {e}")

print("Cargando modelos v4 (Direct MIMO, log_return target)...")

# ── Resolver LFS para todos los modelos v4 ────────────────────────────────
_v4_paths = [
    MODELS_DIR / "direct_gru_v4_5d.keras",
    MODELS_DIR / "direct_lstm_v4_5d.keras",
    MODELS_DIR / "attention_lstm_5d.keras",
    MODELS_DIR / "scaler_v4.pkl",
]
for _p in _v4_paths:
    if _p.exists() and _is_lfs_pointer(_p):
        print(f"[LFS] {_p.name} es puntero — resolviendo...")
        _try_resolve_lfs(_p)

# ── Scaler v4 (8 features, log_return como col 0) ─────────────────────────
try:
    scaler_v4  = joblib.load(MODELS_DIR / "scaler_v4.pkl")
    N_FEATURES = scaler_v4.n_features_in_
    print(f"Scaler v4 cargado ({N_FEATURES} features)")
except Exception as e:
    scaler_v4  = None
    N_FEATURES = 8
    print(f"Advertencia: No se pudo cargar scaler_v4: {e}")

# ── Direct GRU v4 — modelo principal (MAE=142) ────────────────────────────
try:
    from tensorflow import keras
    gru_v4_model = keras.models.load_model(MODELS_DIR / "direct_gru_v4_5d.keras")
    print("Direct GRU v4 (5d) cargado (principal)")
except Exception as e:
    gru_v4_model = None
    print(f"Advertencia: No se pudo cargar GRU v4 5d: {e}")

try:
    from tensorflow import keras as _keras_g10
    gru_v4_10d_model = _keras_g10.models.load_model(MODELS_DIR / "direct_gru_v4_10d.keras")
    print("Direct GRU v4 (10d) cargado")
except Exception as e:
    gru_v4_10d_model = None
    print(f"Advertencia: No se pudo cargar GRU v4 10d: {e}")

# ── Direct LSTM v4 ────────────────────────────────────────────────────────
try:
    from tensorflow import keras as _keras
    lstm_v4_model = _keras.models.load_model(MODELS_DIR / "direct_lstm_v4_5d.keras")
    print("Direct LSTM v4 (5d) cargado")
except Exception as e:
    lstm_v4_model = None
    print(f"Advertencia: No se pudo cargar LSTM v4 5d: {e}")

try:
    from tensorflow import keras as _keras_l10
    lstm_v4_10d_model = _keras_l10.models.load_model(MODELS_DIR / "direct_lstm_v4_10d.keras")
    print("Direct LSTM v4 (10d) cargado")
except Exception as e:
    lstm_v4_10d_model = None
    print(f"Advertencia: No se pudo cargar LSTM v4 10d: {e}")

# ── LSTM + Attention v4 ───────────────────────────────────────────────────
try:
    from tensorflow import keras as _keras2
    attention_model = _keras2.models.load_model(MODELS_DIR / "attention_lstm_5d.keras")
    print("LSTM + Attention v4 cargado")
except Exception as e:
    attention_model = None
    print(f"Advertencia: No se pudo cargar Attention: {e}")

# Alias para compatibilidad con codigo legado que use lstm_model / gru_model / scaler
lstm_model = lstm_v4_model
gru_model  = gru_v4_model
scaler     = scaler_v4

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
_plots_dir = DATA_DIR / "plots"
_plots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(_plots_dir)), name="plots")


# ── Cache en memoria ───────────────────────────────────────────────────────
_cache: dict = {"df": None, "ts": None, "days": None, "source": None}
CACHE_TTL        = 3600   # 1 hora para datos live
_yf_blocked_until: datetime | None = None  # backoff tras rate-limit
YF_BACKOFF        = 1800  # 30 min sin reintentar yfinance tras fallo

# ── Pre-cargar CSV histórico como fallback garantizado ─────────────────────
_RAW_CSV = DATA_DIR / "raw" / "ibex35_raw.csv"
try:
    _static_df = pd.read_csv(_RAW_CSV, index_col=0, parse_dates=True)
    _available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in _static_df.columns]
    _static_df = _static_df[_available].dropna(subset=["Close"])
    _cache["df"]     = _static_df
    _cache["ts"]     = datetime.now()   # caché válida 1h desde ahora
    _cache["days"]   = 3650
    _cache["source"] = "csv"
    print(f"Fallback CSV cargado: {len(_static_df)} dias hasta {_static_df.index[-1].date()}")
except Exception as e:
    print(f"Advertencia: No se pudo cargar CSV fallback: {e}")


# ── Helpers ────────────────────────────────────────────────────────────────
def fetch_recent_ibex(days: int = 90) -> pd.DataFrame:
    """
    Descarga datos del IBEX 35 con caché de 1 hora.
    Fallback garantizado: si yfinance falla usa el CSV histórico bundleado.
    Tras un rate-limit, yfinance se omite 30 min para evitar spam en logs.
    """
    global _yf_blocked_until
    now = datetime.now()

    # Servir caché si está fresco y cubre los días pedidos
    if (_cache["df"] is not None and _cache["ts"] is not None
            and (_cache["days"] or 0) >= days
            and (now - _cache["ts"]).total_seconds() < CACHE_TTL):
        return _cache["df"]

    # Si yfinance está en backoff, ir directo al fallback CSV
    if _yf_blocked_until and now < _yf_blocked_until:
        if _cache["df"] is not None:
            return _cache["df"]

    end       = now
    start     = end - timedelta(days=days + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    for attempt in range(2):   # máx 2 intentos (era 5)
        try:
            data = yf.download("^IBEX", start=start_str, end=end_str,
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # Mantener OHLCV completo — compute_features_v4 necesita High/Low para ATR
            available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
            df = data[available].dropna(subset=["Close"])
            if len(df) > 0:
                _cache["df"]     = df
                _cache["ts"]     = now
                _cache["days"]   = days
                _cache["source"] = "live"
                print(f"yfinance OK: {len(df)} dias hasta {df.index[-1].date()}")
                return df
        except Exception:
            pass
        if attempt < 1:
            time.sleep(2)

    # yfinance falló — activar backoff 30 min y servir CSV
    _yf_blocked_until = now + timedelta(seconds=YF_BACKOFF)
    if _cache["df"] is not None:
        return _cache["df"]

    raise HTTPException(
        status_code=503,
        detail="No se pudo obtener datos. Intenta de nuevo en unos minutos."
    )


def _compute_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula 8 features v4 (log_return target). Requiere OHLCV."""
    from src.preprocessor import compute_features_v4
    return compute_features_v4(df)


def _predict_v4(model, df: pd.DataFrame, n_steps: int = 5) -> list:
    """
    Prediccion MIMO v4: un solo forward pass, reconstruye precios via exp().
    Requiere scaler_v4 y columnas OHLCV en df.
    """
    if model is None or scaler_v4 is None:
        raise HTTPException(status_code=503, detail="Modelo v4 no disponible")

    from src.direct_model import predict_direct_v4

    features = _compute_features_v4(df)
    if len(features) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=400, detail="Datos insuficientes")

    anchor        = float(df["Close"].dropna().iloc[-1])
    last_60_scaled = scaler_v4.transform(features.values[-SEQUENCE_LENGTH:])
    return predict_direct_v4(model, last_60_scaled, scaler_v4, anchor)


def _fmt_predictions(predictions: list, last_price: float,
                      last_date, n_steps: int = 5) -> list:
    """Formatea lista de precios en la estructura estandar de respuesta."""
    future_dates = pd.bdate_range(start=last_date, periods=n_steps + 1)[1:]
    result = []
    for date, price in zip(future_dates, predictions):
        diff = price - last_price
        result.append({
            "fecha":         date.strftime("%Y-%m-%d"),
            "prediccion":    round(price, 2),
            "variacion":     round(diff, 2),
            "variacion_pct": round(diff / last_price * 100, 2),
        })
    return result


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/v2")
def index():
    return FileResponse(str(STATIC_DIR / "index_v2.html"))


@app.get("/v1")
def index_v1():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "gru_v4_loaded":     gru_v4_model is not None,
        "lstm_v4_loaded":    lstm_v4_model is not None,
        "attention_loaded":  attention_model is not None,
        "scaler_v4_loaded":  scaler_v4 is not None,
        "n_features":        N_FEATURES,
        "pipeline":          "v4 (Direct MIMO, log_return target)",
        "best_model":        "Direct GRU v4 (MAE=141.79)",
        "data_source":       _cache.get("source", "none"),
        "cache_date":        _cache["df"].index[-1].strftime("%Y-%m-%d") if _cache["df"] is not None else None,
        "timestamp":         datetime.now().isoformat(),
    }


@app.get("/predict/5days")
def predict_next_5_days():
    """Prediccion principal a 5 dias — Direct GRU v4 (MIMO, log_return, MAE=142)."""
    try:
        df         = fetch_recent_ibex(days=120)
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]

        predictions = _predict_v4(gru_v4_model, df, n_steps=5)
        result      = _fmt_predictions(predictions, last_price, last_date)

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
            "modelo":        "Direct GRU v4 (MIMO)",
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
        df    = fetch_recent_ibex(days=days)
        close = df["Close"].sort_index()
        # Recortar al numero de dias solicitado (importante cuando se sirve desde CSV completo)
        close = close.iloc[-min(days, len(close)):]
        return {
            "dates":  [d.strftime("%Y-%m-%d") for d in close.index],
            "prices": [round(float(p), 2) for p in close.values],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/gru")
def predict_gru():
    """Direct GRU v4 — modelo principal (alias de /predict/5days)."""
    try:
        df         = fetch_recent_ibex(days=120)
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]
        predictions = _predict_v4(gru_v4_model, df, n_steps=5)
        return {
            "modelo":        "Direct GRU v4 (MIMO)",
            "mae_test":      141.79,
            "ultimo_precio": round(last_price, 2),
            "ultima_fecha":  last_date.strftime("%Y-%m-%d"),
            "predicciones":  _fmt_predictions(predictions, last_price, last_date),
            "disclaimer":    "Prediccion orientativa. No constituye asesoramiento financiero.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/lstm")
def predict_lstm():
    """Direct LSTM v4 — MIMO con log_return target (MAE=144)."""
    try:
        df         = fetch_recent_ibex(days=120)
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]
        predictions = _predict_v4(lstm_v4_model, df, n_steps=5)
        return {
            "modelo":        "Direct LSTM v4 (MIMO)",
            "mae_test":      144.35,
            "ultimo_precio": round(last_price, 2),
            "ultima_fecha":  last_date.strftime("%Y-%m-%d"),
            "predicciones":  _fmt_predictions(predictions, last_price, last_date),
            "disclaimer":    "Prediccion orientativa. No constituye asesoramiento financiero.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/attention")
def predict_attention_endpoint():
    """LSTM + Multi-Head Attention v4 (MAE=142)."""
    try:
        from src.attention_model import predict_attention
        df         = fetch_recent_ibex(days=120)
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]

        if attention_model is None or scaler_v4 is None:
            raise HTTPException(status_code=503, detail="Modelo Attention no disponible")

        features = _compute_features_v4(df)
        if len(features) < SEQUENCE_LENGTH:
            raise HTTPException(status_code=400, detail="Datos insuficientes")

        last_60_scaled = scaler_v4.transform(features.values[-SEQUENCE_LENGTH:])
        predictions    = predict_attention(attention_model, last_60_scaled,
                                           scaler_v4, last_price)
        return {
            "modelo":        "LSTM + Attention v4 (MIMO)",
            "mae_test":      141.95,
            "ultimo_precio": round(last_price, 2),
            "ultima_fecha":  last_date.strftime("%Y-%m-%d"),
            "predicciones":  _fmt_predictions(predictions, last_price, last_date),
            "disclaimer":    "Prediccion orientativa. No constituye asesoramiento financiero.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/arima")
def predict_arima():
    """Devuelve prediccion ARIMA a 5 dias desde predictions_5days.csv."""
    try:
        csv_path = DATA_DIR / "predictions_5days.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=503, detail="predictions_5days.csv no encontrado")
        df_pred = pd.read_csv(csv_path)
        df_live = fetch_recent_ibex(days=10)
        last_price = round(float(df_live["Close"].iloc[-1]), 2)
        last_date  = df_live.index[-1].strftime("%Y-%m-%d")
        result = []
        for _, row in df_pred.iterrows():
            pred  = round(float(row["prediccion_arima"]), 2)
            diff  = round(float(row["variacion_arima"]), 2)
            result.append({
                "fecha":         str(row["fecha"]),
                "prediccion":    pred,
                "variacion":     diff,
                "variacion_pct": round(diff / last_price * 100, 2) if last_price else 0,
            })
        return {
            "modelo":        "ARIMA",
            "ultimo_precio": last_price,
            "ultima_fecha":  last_date,
            "predicciones":  result,
            "disclaimer":    "Prediccion orientativa. No constituye asesoramiento financiero.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/validation")
def get_validation():
    """Devuelve validacion multi-periodo desde validation_multiperiod.csv."""
    try:
        csv_path = DATA_DIR / "validation_multiperiod.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=503, detail="validation_multiperiod.csv no encontrado")
        df = pd.read_csv(csv_path)
        return {"periodos": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest")
def get_backtest(days: int = 365):
    """Devuelve los ultimos N dias del backtest completo."""
    try:
        csv_path = DATA_DIR / "backtest_full.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=503, detail="backtest_full.csv no encontrado")
        df = pd.read_csv(csv_path, parse_dates=["fecha"])
        df = df.dropna(subset=["real", "arima"])
        df = df.tail(min(days, len(df)))
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in df["fecha"]],
            "real":  [round(float(v), 2) for v in df["real"]],
            "arima": [round(float(v), 2) for v in df["arima"]],
            "lstm":  [round(float(v), 2) for v in df["lstm"]],
            "gru":   [round(float(v), 2) for v in df["gru"]],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regression")
def get_regression():
    """Devuelve R², DW y betas por modelo desde regression_summary.csv."""
    try:
        csv_path = DATA_DIR / "regression_summary.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=503, detail="regression_summary.csv no encontrado")
        df = pd.read_csv(csv_path)
        df_completo = df[df["periodo"] == "Completo"]
        result = []
        for _, row in df_completo.iterrows():
            result.append({
                "modelo": row["modelo"],
                "r2":     round(float(row["r2"]), 5),
                "r2_adj": round(float(row["r2_adj"]), 5),
                "dw":     round(float(row["durbin_watson"]), 4),
                "sigma":  round(float(row["sigma"]), 2),
                "beta1":  round(float(row["beta1"]), 6),
            })
        return {"regresion": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/horizon")
def predict_horizon(days: int = 5):
    """
    Prediccion a N dias habiles (1-10): GRU v4 + LSTM v4 + Attention + ARIMA.
    GRU v4 es el modelo principal (MAE=142). ARIMA es baseline estadistico.
    """
    days = max(1, min(days, 10))
    try:
        import warnings
        from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
        from src.attention_model import predict_attention

        df         = fetch_recent_ibex(days=150)
        last_price = float(df["Close"].iloc[-1])
        last_date  = df.index[-1]
        future_dates = [d.strftime("%Y-%m-%d")
                        for d in pd.bdate_range(start=last_date, periods=days + 1)[1:]]

        # ── ARIMA: baseline estadistico ───────────────────────
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            series    = df["Close"].values.astype(float)
            arima_fit = ARIMAModel(series, order=(5, 1, 0)).fit()
            arima_fc  = arima_fit.forecast(steps=days)
        arima_preds = [round(float(v), 2) for v in arima_fc]

        # ── Preparar features v4 ──────────────────────────────
        features_v4    = _compute_features_v4(df)
        last_60_scaled = scaler_v4.transform(features_v4.values[-SEQUENCE_LENGTH:]) \
                         if scaler_v4 and len(features_v4) >= SEQUENCE_LENGTH else None

        # ── GRU v4 — usa modelo 10d si days > 5 ──────────────
        gru_preds = None
        _gru_m = (gru_v4_10d_model if days > 5 and gru_v4_10d_model else gru_v4_model)
        if _gru_m and last_60_scaled is not None:
            from src.direct_model import predict_direct_v4
            all_gru = predict_direct_v4(_gru_m, last_60_scaled, scaler_v4, last_price)
            gru_preds = [round(v, 2) for v in all_gru[:days]]

        # ── LSTM v4 — usa modelo 10d si days > 5 ────────────
        lstm_preds = None
        _lstm_m = (lstm_v4_10d_model if days > 5 and lstm_v4_10d_model else lstm_v4_model)
        if _lstm_m and last_60_scaled is not None:
            from src.direct_model import predict_direct_v4 as _pdv4
            all_lstm = _pdv4(_lstm_m, last_60_scaled, scaler_v4, last_price)
            lstm_preds = [round(v, 2) for v in all_lstm[:days]]

        # ── Attention v4 ──────────────────────────────────────
        attn_preds = None
        if attention_model and last_60_scaled is not None:
            attn_preds = [round(v, 2) for v in
                          predict_attention(attention_model, last_60_scaled,
                                            scaler_v4, last_price)[:days]]

        result = []
        for i, fecha in enumerate(future_dates):
            ap  = arima_preds[i] if arima_preds and i < len(arima_preds) else None
            gp  = gru_preds[i]   if gru_preds   and i < len(gru_preds)   else None
            lp  = lstm_preds[i]  if lstm_preds   and i < len(lstm_preds)  else None
            ap2 = attn_preds[i]  if attn_preds   and i < len(attn_preds)  else None
            result.append({
                "fecha":          fecha,
                "gru_v4":         gp,
                "lstm_v4":        lp,
                "attention_v4":   ap2,
                "arima":          ap,
                "gru_var_pct":    round((gp  - last_price) / last_price * 100, 2) if gp  else None,
                "lstm_var_pct":   round((lp  - last_price) / last_price * 100, 2) if lp  else None,
                "attn_var_pct":   round((ap2 - last_price) / last_price * 100, 2) if ap2 else None,
                "arima_var_pct":  round((ap  - last_price) / last_price * 100, 2) if ap else None,
            })

        return {
            "ultimo_precio": round(last_price, 2),
            "ultima_fecha":  last_date.strftime("%Y-%m-%d"),
            "days":          days,
            "modelos":       ["Direct GRU v4", "Direct LSTM v4", "LSTM+Attention v4", "ARIMA"],
            "predicciones":  result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    """Devuelve metricas comparativas ARIMA vs LSTM vs GRU."""
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
