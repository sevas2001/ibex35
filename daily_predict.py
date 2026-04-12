"""
daily_predict.py — Ejecutar una vez al dia (después del cierre del mercado)

  python daily_predict.py

Hace la prediccion LSTM, GRU y ARIMA para manana, actualiza precios reales
de dias anteriores y muestra el resumen de accuracy acumulado.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

from src.prediction_logger import save_prediction, update_with_real_prices, get_accuracy_summary

MODELS_DIR      = Path("models")
DATA_DIR        = Path("data")
SEQUENCE_LENGTH = 60
N_FEATURES      = 1

# ── Cargar modelos LSTM y GRU ─────────────────────────────────────────────────
print("Cargando modelos LSTM y GRU...")
lstm_model = keras.models.load_model(MODELS_DIR / "lstm_model.keras")
gru_model  = keras.models.load_model(MODELS_DIR / "gru_model.keras")
scaler     = joblib.load(MODELS_DIR / "scaler.pkl")
print("OK")

# ── Descargar datos recientes (yfinance) ──────────────────────────────────────
print("Descargando datos IBEX 35...")
end   = datetime.now() + timedelta(days=1)   # end es exclusivo en yfinance
start = end - timedelta(days=121)
data  = yf.download("^IBEX", start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=True, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df        = data[["Close"]].dropna()
last_date  = df.index[-1]
last_price = float(df["Close"].iloc[-1])
print(f"Ultimo precio: {last_price:.2f} pts  ({last_date.date()})")

# ── Prediccion LSTM ───────────────────────────────────────────────────────────
last_60  = df.values[-SEQUENCE_LENGTH:]
scaled   = scaler.transform(last_60)
seq      = scaled.tolist()

preds_scaled = []
for _ in range(5):
    X = np.array(seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
    p = float(lstm_model.predict(X, verbose=0)[0, 0])
    preds_scaled.append(p)
    seq.append([p])

dummy = np.zeros((len(preds_scaled), N_FEATURES))
dummy[:, 0] = preds_scaled
lstm_preds = scaler.inverse_transform(dummy)[:, 0].tolist()

# ── Prediccion GRU ────────────────────────────────────────────────────────────
seq_gru = scaled.tolist()
gru_scaled = []
for _ in range(5):
    X = np.array(seq_gru[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
    p = float(gru_model.predict(X, verbose=0)[0, 0])
    gru_scaled.append(p)
    seq_gru.append([p])

dummy[:, 0] = gru_scaled
gru_preds = scaler.inverse_transform(dummy)[:, 0].tolist()

# ── Prediccion ARIMA(5,1,0) ───────────────────────────────────────────────────
print("Ajustando ARIMA(5,1,0)...")
hist_path = DATA_DIR / "raw" / "ibex35_full.csv"
if hist_path.exists():
    hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)
    arima_series = hist["Close"].values[-200:].astype(float)
else:
    # fallback: usar los mismos datos descargados de yfinance
    arima_series = df["Close"].values.astype(float)

arima_fit      = ARIMAModel(arima_series, order=(5, 1, 0)).fit()
arima_forecast = arima_fit.forecast(steps=5)
arima_preds    = [float(v) for v in arima_forecast]

# ── Imprimir predicciones ─────────────────────────────────────────────────────
future_dates = pd.bdate_range(start=last_date, periods=6)[1:]

print(f"\nPredicciones proximos 5 dias (base: {last_price:.2f} pts):")
print(f"  {'Fecha':<12}  {'ARIMA':>10}  {'LSTM':>10}  {'GRU':>10}")
print("  " + "-"*46)
for date, ap, lp, gp in zip(future_dates, arima_preds, lstm_preds, gru_preds):
    da = ap - last_price
    dl = lp - last_price
    dg = gp - last_price
    print(f"  {str(date.date()):<12}  "
          f"{ap:>8.2f} ({'+' if da>=0 else ''}{da:.0f})  "
          f"{lp:>8.2f} ({'+' if dl>=0 else ''}{dl:.0f})  "
          f"{gp:>8.2f} ({'+' if dg>=0 else ''}{dg:.0f})")

# ── Guardar prediccion D+1 en el log ──────────────────────────────────────────
save_prediction(
    precio_base      = last_price,
    prediccion_d1    = lstm_preds[0],
    prediccion_arima = arima_preds[0],
    fecha            = last_date.strftime("%Y-%m-%d"),
)

# ── Guardar tabla de 5 dias ───────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "fecha":            [d.date() for d in future_dates],
    "prediccion_arima": [round(p, 2) for p in arima_preds],
    "prediccion_lstm":  [round(p, 2) for p in lstm_preds],
    "prediccion_gru":   [round(p, 2) for p in gru_preds],
    "variacion_arima":  [round(p - last_price, 2) for p in arima_preds],
    "variacion_lstm":   [round(p - last_price, 2) for p in lstm_preds],
    "variacion_gru":    [round(p - last_price, 2) for p in gru_preds],
})
pred_df.to_csv(DATA_DIR / "predictions_5days.csv", index=False)

# ── Actualizar precios reales de dias anteriores ───────────────────────────────
print("\nActualizando precios reales en el log...")
update_with_real_prices()

# ── Resumen de accuracy ────────────────────────────────────────────────────────
summary = get_accuracy_summary()
print("\n" + "=" * 55)
print("  ACCURACY ACUMULADO")
print("=" * 55)
print(f"  Total predicciones:  {summary['total_predicciones']}")
print(f"  Evaluadas:           {summary['evaluadas']}")
if summary["direction_accuracy_pct"] is not None:
    print(f"\n  LSTM:")
    print(f"    Direction accuracy:  {summary['direction_accuracy_pct']}%")
    print(f"    MAE:                 {summary['mae']} pts")
    print(f"    RMSE:                {summary['rmse']} pts")
if summary.get("direction_arima_pct") is not None:
    print(f"\n  ARIMA:")
    print(f"    Direction accuracy:  {summary['direction_arima_pct']}%")
    print(f"    MAE:                 {summary['mae_arima']} pts")

if summary["historial"]:
    print("\nUltimas 5 evaluadas:")
    for r in summary["historial"][:5]:
        ok_l = "OK" if r.get("direction_correct") == 1 else "FAIL"
        ok_a = "OK" if r.get("direction_arima") == 1 else ("FAIL" if r.get("direction_arima") == 0 else "---")
        arima_err = f"err_arima={r.get('error_arima', '---')}" if r.get("error_arima") else ""
        print(f"  {r['fecha_prediccion']}  "
              f"lstm={r['prediccion_d1']}  real={r['real_d1']}  err={r['error_abs']}  [LSTM:{ok_l}]  "
              f"arima={r.get('prediccion_arima', '---')}  {arima_err}  [ARIMA:{ok_a}]")

print("\nLog guardado en data/prediction_log.csv")
print("Predicciones 5 dias en data/predictions_5days.csv")
