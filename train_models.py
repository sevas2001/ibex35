"""
IBEX 35 — Fase 2: Entrenamiento de Modelos ARIMA + LSTM + GRU (univariado)
Comparativa triple: ARIMA(5,1,0) baseline vs LSTM vs GRU (Cho et al., 2014)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.data_loader   import load_ibex35
from src.preprocessor  import TRAIN_RATIO, SEQUENCE_LENGTH, preprocess, compute_features
from src.arima_model   import test_stationarity, plot_acf_pacf, train_arima, evaluate_arima
from src.lstm_model    import train_lstm, plot_training_history, evaluate_lstm, predict_next_5_days
from src.gru_model     import train_gru, plot_gru_training_history, evaluate_gru, predict_next_5_days_gru
from src.evaluator     import compare_models, plot_predictions, plot_metrics_bar

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")


def main():
    print("=" * 55)
    print("  IBEX 35 — Fase 2: Modelos ARIMA + LSTM + GRU")
    print("=" * 55)

    # Cargar datos
    data  = load_ibex35()
    close = data["Close"]

    split_idx    = int(len(close) * TRAIN_RATIO)
    train_series = close.iloc[:split_idx]
    test_series  = close.iloc[split_idx:]
    test_index   = test_series.index

    # Preprocesar (genera preprocessed_data.npz y scaler.pkl)
    prep = preprocess(data)
    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_test  = prep["X_test"]
    y_test  = prep["y_test"]
    scaler  = prep["scaler"]

    # ── ARIMA ──────────────────────────────────────────────
    print("\n--- ARIMA ---")
    test_stationarity(train_series)
    plot_acf_pacf(train_series)
    arima_fitted  = train_arima(train_series, order=(5, 1, 0))
    arima_results = evaluate_arima(arima_fitted, train_series, test_series)

    # ── LSTM ───────────────────────────────────────────────
    print("\n--- LSTM (univariado) ---")
    print(f"Input shape: {X_train.shape}  (samples, seq_len, features)")
    lstm_model, history = train_lstm(X_train, y_train)
    plot_training_history(history)
    lstm_results = evaluate_lstm(lstm_model, X_test, y_test, scaler)

    # ── GRU ────────────────────────────────────────────────
    print("\n--- GRU (Cho et al., 2014) ---")
    print(f"Input shape: {X_train.shape}  (samples, seq_len, features)")
    gru_model, gru_history = train_gru(X_train, y_train)
    plot_gru_training_history(gru_history)
    gru_results = evaluate_gru(gru_model, X_test, y_test, scaler)

    # ── COMPARATIVA ────────────────────────────────────────
    comparison = compare_models(arima_results, lstm_results, test_index, gru_results)
    plot_predictions(
        lstm_results["actuals"],
        arima_results["predictions"],
        lstm_results["predictions"],
        test_index,
        gru_preds=gru_results["predictions"],
    )
    plot_metrics_bar(comparison)
    comparison.to_csv(DATA_DIR / "metrics_comparison.csv", index=False)

    # ── PREDICCION A 5 DIAS (LSTM) ─────────────────────────
    print("\n--- Prediccion a 5 dias ---")
    features = compute_features(data)
    last_60_raw    = features.values[-SEQUENCE_LENGTH:]
    last_60_scaled = scaler.transform(last_60_raw)
    predictions_5d = predict_next_5_days(lstm_model, last_60_scaled, scaler)

    # GRU 5 days
    gru_5d = predict_next_5_days_gru(gru_model, last_60_scaled, scaler)

    last_date = close.index[-1]
    dates_5d  = pd.bdate_range(start=last_date, periods=6)[1:]
    print(f"\nUltimo precio real: {close.iloc[-1]:.2f} pts ({last_date.date()})")
    print("\nPrediccion proximos 5 dias habiles (LSTM vs GRU):")
    for date, lstm_p, gru_p in zip(dates_5d, predictions_5d, gru_5d):
        diff_l = lstm_p - close.iloc[-1]
        diff_g = gru_p  - close.iloc[-1]
        sign_l = "+" if diff_l >= 0 else ""
        sign_g = "+" if diff_g >= 0 else ""
        print(f"  {date.date()}  LSTM: {lstm_p:.2f} ({sign_l}{diff_l:.2f})  "
              f"GRU: {gru_p:.2f} ({sign_g}{diff_g:.2f})")

    pred_df = pd.DataFrame({
        "fecha":            [d.date() for d in dates_5d],
        "prediccion_lstm":  [round(p, 2) for p in predictions_5d],
        "prediccion_gru":   [round(p, 2) for p in gru_5d],
        "variacion_lstm":   [round(p - close.iloc[-1], 2) for p in predictions_5d],
        "variacion_gru":    [round(p - close.iloc[-1], 2) for p in gru_5d],
    })
    pred_df.to_csv(DATA_DIR / "predictions_5days.csv", index=False)
    print("\nPredicciones guardadas en data/predictions_5days.csv")
    print("\nFase 2 completada.")


if __name__ == "__main__":
    main()
