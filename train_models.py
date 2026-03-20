"""
IBEX 35 — Fase 2: Entrenamiento de Modelos ARIMA + LSTM
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.data_loader import load_ibex35
from src.preprocessor import TRAIN_RATIO, SEQUENCE_LENGTH
from src.arima_model import test_stationarity, plot_acf_pacf, train_arima, evaluate_arima
from src.lstm_model import train_lstm, plot_training_history, evaluate_lstm, predict_next_5_days
from src.evaluator import compare_models, plot_predictions, plot_metrics_bar

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def main():
    print("=" * 50)
    print("  IBEX 35 — Fase 2: Modelos")
    print("=" * 50)

    # Cargar datos y preprocessed
    data = load_ibex35()
    close = data["Close"]

    split_idx = int(len(close) * TRAIN_RATIO)
    train_series = close.iloc[:split_idx]
    test_series = close.iloc[split_idx:]
    test_index = test_series.index

    # Cargar datos preprocesados para LSTM
    npz = np.load(DATA_DIR / "preprocessed_data.npz")
    X_train = npz["X_train"]
    y_train = npz["y_train"]
    X_test = npz["X_test"]
    y_test = npz["y_test"]
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    # ── ARIMA ──────────────────────────────────────────────
    print("\n--- ARIMA ---")
    test_stationarity(train_series)
    plot_acf_pacf(train_series)
    arima_fitted = train_arima(train_series, order=(5, 1, 0))
    arima_results = evaluate_arima(arima_fitted, train_series, test_series)

    # ── LSTM ───────────────────────────────────────────────
    print("\n--- LSTM ---")
    lstm_model, history = train_lstm(X_train, y_train)
    plot_training_history(history)
    lstm_results = evaluate_lstm(lstm_model, X_test, y_test, scaler)

    # ── COMPARATIVA ────────────────────────────────────────
    comparison = compare_models(arima_results, lstm_results, test_index)
    plot_predictions(
        lstm_results["actuals"],
        arima_results["predictions"],
        lstm_results["predictions"],
        test_index,
    )
    plot_metrics_bar(comparison)
    comparison.to_csv(DATA_DIR / "metrics_comparison.csv", index=False)

    # ── PREDICCION A 5 DIAS ────────────────────────────────
    print("\n--- Prediccion a 5 dias ---")
    last_60_scaled = scaler.transform(
        close.values[-SEQUENCE_LENGTH:].reshape(-1, 1)
    )
    predictions_5d = predict_next_5_days(lstm_model, last_60_scaled, scaler)

    last_date = close.index[-1]
    dates_5d = pd.bdate_range(start=last_date, periods=6)[1:]  # 5 dias habiles
    print(f"\nUltimo precio real: {close.iloc[-1]:.2f} pts ({last_date.date()})")
    print("\nPrediccion proximos 5 dias habiles:")
    for date, price in zip(dates_5d, predictions_5d):
        diff = price - close.iloc[-1]
        sign = "+" if diff >= 0 else ""
        print(f"  {date.date()}  ->  {price:.2f} pts  ({sign}{diff:.2f})")

    # Guardar predicciones
    pred_df = pd.DataFrame({
        "fecha": [d.date() for d in dates_5d],
        "prediccion": [round(p, 2) for p in predictions_5d],
        "variacion": [round(p - close.iloc[-1], 2) for p in predictions_5d],
        "variacion_pct": [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 2)
                          for p in predictions_5d],
    })
    pred_df.to_csv(DATA_DIR / "predictions_5days.csv", index=False)
    print("\nPredicciones guardadas en data/predictions_5days.csv")
    print("\nFase 2 completada.")


if __name__ == "__main__":
    main()
