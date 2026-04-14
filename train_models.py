"""
IBEX 35 — Fase 2: Entrenamiento de Modelos ARIMA + LSTM + GRU (univariado)
Comparativa triple: ARIMA(5,1,0) baseline vs LSTM vs GRU (Cho et al., 2014)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.data_loader   import load_ibex35
from src.preprocessor  import (TRAIN_RATIO, SEQUENCE_LENGTH,
                                preprocess, preprocess_v2, preprocess_v3, preprocess_v4,
                                compute_features, compute_features_v2, compute_features_v3,
                                compute_features_v4)
from src.arima_model   import test_stationarity, plot_acf_pacf, train_arima, evaluate_arima
from src.lstm_model    import train_lstm, plot_training_history, evaluate_lstm, predict_next_5_days, predict_next_n_days_v2
from src.gru_model     import train_gru, plot_gru_training_history, evaluate_gru, predict_next_5_days_gru, predict_next_n_days_v2_gru
from src.hybrid_model  import train_hybrid, evaluate_hybrid, plot_hybrid_training, predict_next_n_days_hybrid
from src.direct_model  import (create_sequences_direct, train_direct, evaluate_direct,
                                predict_direct, plot_direct_training,
                                create_sequences_direct_v4, train_direct_v4, evaluate_direct_v4,
                                predict_direct_v4)
from src.attention_model import (build_lstm_attention, train_attention, predict_attention,
                                  plot_attention_training)
from src.evaluator     import compare_models, plot_predictions, plot_metrics_bar
from src.backtester    import run_backtest, plot_backtest

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")


def main():
    print("=" * 55)
    print("  IBEX 35 — Fase 2: Modelos ARIMA + LSTM + GRU")
    print("=" * 55)

    # Cargar datos — historico completo desde 1993
    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos cargados: {len(close)} dias | {close.index[0].date()} hasta {close.index[-1].date()}")

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

    # ── BACKTESTING dia a dia — serie COMPLETA desde 1993 ────
    print("\n--- Backtesting completo (desde 1993) ---")
    full_close_arr = close.values.astype(float)
    bt_df = run_backtest(
        full_close   = full_close_arr,
        full_index   = close.index,
        train_size   = split_idx,
        scaler       = scaler,
        lstm_model   = lstm_model,
        gru_model    = gru_model,
        arima_order  = (5, 1, 0),
        full_history = True,
    )
    train_cutoff = close.index[split_idx]
    plot_backtest(bt_df, train_cutoff_date=train_cutoff)

    print("\nFase 2 completada.")


def train_v2():
    """
    Pipeline v2: LSTM + GRU con 7 indicadores tecnicos.
    ARIMA se mantiene como baseline estadistico (sin cambios).
    Guarda: lstm_model_v2.keras, gru_model_v2.keras, scaler_v2.pkl
    """
    print("=" * 60)
    print("  IBEX 35 — Entrenamiento v2: LSTM + GRU con 7 features")
    print("=" * 60)

    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos cargados: {len(close)} dias | {close.index[0].date()} — {close.index[-1].date()}")

    split_idx    = int(len(close) * TRAIN_RATIO)
    train_series = close.iloc[:split_idx]
    test_series  = close.iloc[split_idx:]
    test_index   = test_series.index

    # Preprocesar con 7 features (genera preprocessed_data_v2.npz y scaler_v2.pkl)
    prep = preprocess_v2(data)
    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_test  = prep["X_test"]
    y_test  = prep["y_test"]
    scaler  = prep["scaler"]
    print(f"\nInput shape: {X_train.shape}  (samples, seq_len=60, n_features=7)")

    # ── ARIMA (baseline — sin cambios) ────────────────────────────────────────
    print("\n--- ARIMA (5,1,0) — baseline estadistico ---")
    arima_fitted  = train_arima(train_series, order=(5, 1, 0))
    arima_results = evaluate_arima(arima_fitted, train_series, test_series)

    # ── LSTM v2 ───────────────────────────────────────────────────────────────
    print("\n--- LSTM v2 (7 features) ---")
    lstm_model, lstm_history = train_lstm(X_train, y_train)
    plot_training_history(lstm_history)
    lstm_results = evaluate_lstm(lstm_model, X_test, y_test, scaler)

    # ── GRU v2 ────────────────────────────────────────────────────────────────
    print("\n--- GRU v2 (7 features) ---")
    gru_model, gru_history = train_gru(X_train, y_train)
    plot_gru_training_history(gru_history)
    gru_results = evaluate_gru(gru_model, X_test, y_test, scaler)

    # ── COMPARATIVA ───────────────────────────────────────────────────────────
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
    print("\nMetricas v2 guardadas en data/metrics_comparison.csv")

    # ── PREDICCION A 5 DIAS (v2) ──────────────────────────────────────────────
    print("\n--- Prediccion a 5 dias (v2) ---")
    features_v2    = compute_features_v2(data)
    last_60_raw    = features_v2.values[-SEQUENCE_LENGTH:]
    last_60_scaled = scaler.transform(last_60_raw)

    lstm_5d = predict_next_n_days_v2(lstm_model, last_60_scaled, scaler, n_steps=5)
    gru_5d  = predict_next_n_days_v2_gru(gru_model, last_60_scaled, scaler, n_steps=5)

    # ARIMA 5 dias (baseline)
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
    arima_fit      = ARIMAModel(train_series.values.astype(float), order=(5, 1, 0)).fit()
    arima_5d       = [float(v) for v in arima_fit.forecast(steps=5)]

    last_date = close.index[-1]
    dates_5d  = pd.bdate_range(start=last_date, periods=6)[1:]
    print(f"\nUltimo precio real: {close.iloc[-1]:.2f} pts ({last_date.date()})")
    print(f"\n{'Fecha':<14} {'ARIMA':>10} {'LSTM v2':>10} {'GRU v2':>10}")
    print("-" * 46)
    for date, ap, lp, gp in zip(dates_5d, arima_5d, lstm_5d, gru_5d):
        print(f"{str(date.date()):<14} {ap:>10.2f} {lp:>10.2f} {gp:>10.2f}")

    pred_df = pd.DataFrame({
        "fecha":            [d.date() for d in dates_5d],
        "prediccion_arima": [round(p, 2) for p in arima_5d],
        "prediccion_lstm":  [round(p, 2) for p in lstm_5d],
        "prediccion_gru":   [round(p, 2) for p in gru_5d],
        "variacion_arima":  [round(p - close.iloc[-1], 2) for p in arima_5d],
        "variacion_lstm":   [round(p - close.iloc[-1], 2) for p in lstm_5d],
        "variacion_gru":    [round(p - close.iloc[-1], 2) for p in gru_5d],
        "variacion_pct_arima": [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 3) for p in arima_5d],
        "variacion_pct_lstm":  [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 3) for p in lstm_5d],
        "variacion_pct_gru":   [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 3) for p in gru_5d],
    })
    pred_df.to_csv(DATA_DIR / "predictions_5days.csv", index=False)
    print("\nPredicciones v2 guardadas en data/predictions_5days.csv")

    print("\nEntrenamiento v2 completado.")
    print(f"  LSTM v2 — MAE: {lstm_results['mae']:.2f} | RMSE: {lstm_results['rmse']:.2f}")
    print(f"  GRU  v2 — MAE: {gru_results['mae']:.2f} | RMSE: {gru_results['rmse']:.2f}")


def train_v3():
    """
    Pipeline v3: LSTM + GRU con 5 features seleccionadas + eventos historicos IBEX 35.
    Features: close, log_return, ma50_ratio, crisis_regime, crisis_decay.
    Mejoras vs v2: menos features (sin redundancia), eventos macro, patience x3.
    Guarda: lstm_model_v3.keras, gru_model_v3.keras, scaler_v3.pkl
    """
    print("=" * 65)
    print("  IBEX 35 — Entrenamiento v3: 5 features + eventos historicos")
    print("=" * 65)

    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos cargados: {len(close)} dias | {close.index[0].date()} — {close.index[-1].date()}")

    split_idx    = int(len(close) * TRAIN_RATIO)
    train_series = close.iloc[:split_idx]
    test_series  = close.iloc[split_idx:]
    test_index   = test_series.index

    # Preprocesar con 5 features v3
    prep    = preprocess_v3(data)
    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_test  = prep["X_test"]
    y_test  = prep["y_test"]
    scaler  = prep["scaler"]
    print(f"\nInput shape: {X_train.shape}  (samples, seq_len=60, n_features=5)")

    # ── ARIMA (baseline sin cambios) ──────────────────────────────────────────
    print("\n--- ARIMA (5,1,0) — baseline estadistico ---")
    arima_fitted  = train_arima(train_series, order=(5, 1, 0))
    arima_results = evaluate_arima(arima_fitted, train_series, test_series)

    # ── LSTM v3 ───────────────────────────────────────────────────────────────
    print("\n--- LSTM v3 (5 features + eventos historicos) ---")
    lstm_model, lstm_history = train_lstm(X_train, y_train)
    plot_training_history(lstm_history)
    lstm_results = evaluate_lstm(lstm_model, X_test, y_test, scaler)

    # ── GRU v3 ────────────────────────────────────────────────────────────────
    print("\n--- GRU v3 (5 features + eventos historicos) ---")
    gru_model, gru_history = train_gru(X_train, y_train)
    plot_gru_training_history(gru_history)
    gru_results = evaluate_gru(gru_model, X_test, y_test, scaler)

    # ── COMPARATIVA ───────────────────────────────────────────────────────────
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
    print("\nMetricas v3 guardadas en data/metrics_comparison.csv")

    # ── PREDICCION A 5 DIAS ───────────────────────────────────────────────────
    print("\n--- Prediccion a 5 dias (v3) ---")
    features_v3    = compute_features_v3(data)
    last_60_raw    = features_v3.values[-SEQUENCE_LENGTH:]
    last_60_scaled = scaler.transform(last_60_raw)

    lstm_5d = predict_next_n_days_v2(lstm_model, last_60_scaled, scaler, n_steps=5)
    gru_5d  = predict_next_n_days_v2_gru(gru_model, last_60_scaled, scaler, n_steps=5)

    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
    arima_fit = ARIMAModel(train_series.values.astype(float), order=(5, 1, 0)).fit()
    arima_5d  = [float(v) for v in arima_fit.forecast(steps=5)]

    last_date = close.index[-1]
    dates_5d  = pd.bdate_range(start=last_date, periods=6)[1:]
    print(f"\nUltimo precio real: {close.iloc[-1]:.2f} pts ({last_date.date()})")
    print(f"\n{'Fecha':<14} {'ARIMA':>10} {'LSTM v3':>10} {'GRU v3':>10}")
    print("-" * 46)
    for date, ap, lp, gp in zip(dates_5d, arima_5d, lstm_5d, gru_5d):
        print(f"{str(date.date()):<14} {ap:>10.2f} {lp:>10.2f} {gp:>10.2f}")

    pred_df = pd.DataFrame({
        "fecha":               [d.date() for d in dates_5d],
        "prediccion_arima":    [round(p, 2) for p in arima_5d],
        "prediccion_lstm":     [round(p, 2) for p in lstm_5d],
        "prediccion_gru":      [round(p, 2) for p in gru_5d],
        "variacion_arima":     [round(p - close.iloc[-1], 2) for p in arima_5d],
        "variacion_lstm":      [round(p - close.iloc[-1], 2) for p in lstm_5d],
        "variacion_gru":       [round(p - close.iloc[-1], 2) for p in gru_5d],
        "variacion_pct_arima": [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 3) for p in arima_5d],
        "variacion_pct_lstm":  [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 3) for p in lstm_5d],
        "variacion_pct_gru":   [round((p - close.iloc[-1]) / close.iloc[-1] * 100, 3) for p in gru_5d],
    })
    pred_df.to_csv(DATA_DIR / "predictions_5days.csv", index=False)

    print("\n" + "=" * 65)
    print("  RESULTADOS FINALES v3")
    print("=" * 65)
    print(f"  ARIMA(5,1,0)  MAE: {arima_results['mae']:.2f} | RMSE: {arima_results['rmse']:.2f}")
    print(f"  LSTM v3       MAE: {lstm_results['mae']:.2f} | RMSE: {lstm_results['rmse']:.2f}")
    print(f"  GRU  v3       MAE: {gru_results['mae']:.2f} | RMSE: {gru_results['rmse']:.2f}")
    print("\nEntrenamiento v3 completado.")


def train_hybrid_models():
    """
    Entrena dos variantes del modelo Hybrid LSTM-GRU (Plan B):
      - Univariada (Close): usa preprocess() + scaler.pkl  -> hybrid_model.keras
      - Multivariada (5f):  usa preprocess_v3() + scaler_v3.pkl -> hybrid_model_v3.keras
    Imprime tabla comparativa con todos los modelos anteriores.
    """
    print("=" * 65)
    print("  IBEX 35 — Plan B: Hybrid LSTM-GRU")
    print("=" * 65)

    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos cargados: {len(close)} dias | {close.index[0].date()} — {close.index[-1].date()}")

    split_idx    = int(len(close) * TRAIN_RATIO)
    train_series = close.iloc[:split_idx]
    test_series  = close.iloc[split_idx:]
    test_index   = test_series.index

    # ── Variante 1: univariada (Close) ────────────────────────────────────────
    print("\n[1/2] Hybrid univariado (Close, n_features=1)")
    prep1   = preprocess(data)
    scaler1 = prep1["scaler"]

    hyb1_model, hyb1_history = train_hybrid(prep1["X_train"], prep1["y_train"])
    plot_hybrid_training(hyb1_history)
    hyb1_results = evaluate_hybrid(hyb1_model, prep1["X_test"], prep1["y_test"], scaler1)

    # ── Variante 2: multivariada v3 (5 features) ──────────────────────────────
    print("\n[2/2] Hybrid v3 (5 features + crisis events, n_features=5)")
    prep3   = preprocess_v3(data)
    scaler3 = prep3["scaler"]

    hyb3_model, hyb3_history = train_hybrid(prep3["X_train"], prep3["y_train"])
    hyb3_results = evaluate_hybrid(hyb3_model, prep3["X_test"], prep3["y_test"], scaler3)

    # ── Prediccion a 5 dias ───────────────────────────────────────────────────
    print("\n--- Prediccion a 5 dias ---")
    features1    = compute_features(data)
    last_60_raw1 = features1.values[-SEQUENCE_LENGTH:]
    last_60_sc1  = scaler1.transform(last_60_raw1)
    hyb1_5d = predict_next_n_days_hybrid(hyb1_model, last_60_sc1, scaler1)

    features3    = compute_features_v3(data)
    last_60_raw3 = features3.values[-SEQUENCE_LENGTH:]
    last_60_sc3  = scaler3.transform(last_60_raw3)
    hyb3_5d = predict_next_n_days_hybrid(hyb3_model, last_60_sc3, scaler3)

    last_date = close.index[-1]
    dates_5d  = pd.bdate_range(start=last_date, periods=6)[1:]
    print(f"\nUltimo precio real: {close.iloc[-1]:.2f} pts ({last_date.date()})")
    print(f"\n{'Fecha':<14} {'Hybrid v1':>12} {'Hybrid v3':>12}")
    print("-" * 40)
    for date, h1, h3 in zip(dates_5d, hyb1_5d, hyb3_5d):
        print(f"{str(date.date()):<14} {h1:>12.2f} {h3:>12.2f}")

    # ── Tabla comparativa final ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TABLA COMPARATIVA COMPLETA (todos los modelos)")
    print("=" * 65)
    print(f"  {'Modelo':<25} {'MAE':>8} {'RMSE':>8}")
    print("  " + "-" * 43)
    print(f"  {'ARIMA(5,1,0) baseline':<25} {'82':>8} {'118':>8}  (referencia)")
    print(f"  {'LSTM v1 (univariado)':<25} {'126':>8} {'177':>8}")
    print(f"  {'GRU v1  (univariado)':<25} {'110':>8} {'155':>8}  <- mejor individual")
    print(f"  {'LSTM v2 (7 features)':<25} {'131':>8} {'194':>8}")
    print(f"  {'GRU v2  (7 features)':<25} {'152':>8} {'226':>8}")
    print(f"  {'LSTM v3 (5f+crisis)':<25} {'160':>8} {'234':>8}")
    print(f"  {'GRU v3  (5f+crisis)':<25} {'129':>8} {'191':>8}")
    print(f"  {'Hybrid LSTM-GRU v1':<25} {hyb1_results['mae']:>8.2f} {hyb1_results['rmse']:>8.2f}  <- NUEVO")
    print(f"  {'Hybrid LSTM-GRU v3':<25} {hyb3_results['mae']:>8.2f} {hyb3_results['rmse']:>8.2f}  <- NUEVO")
    print("=" * 65)

    gru_v1_mae = 110
    if hyb1_results["mae"] < gru_v1_mae:
        best_mae = min(hyb1_results["mae"], hyb3_results["mae"])
        improvement = (gru_v1_mae - best_mae) / gru_v1_mae * 100
        print(f"\n  RESULTADO: El hibrido MEJORA al GRU v1 en {improvement:.1f}%")
        print(f"  -> Actualizar api.py y daily_predict.py con el modelo hibrido.")
    else:
        print(f"\n  RESULTADO: El hibrido NO supera al GRU v1 (MAE={gru_v1_mae}).")
        print(f"  -> GRU v1 sigue siendo el modelo de produccion.")
        print(f"  -> Resultado valido: demuestra que la arquitectura simple es suficiente.")

    print("\nEntrenamiento hibrido completado.")


def train_direct_models():
    """
    Entrena modelos Direct MIMO (LSTM + GRU) con n_steps=5 para produccion
    y n_steps=11 para comparativa abril 2026.
    Compara MAE por paso con los modelos recursivos v1.
    """
    print("=" * 65)
    print("  IBEX 35 — Direct MIMO: LSTM + GRU (n_steps=5 y n_steps=11)")
    print("=" * 65)

    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos: {len(close)} dias | {close.index[0].date()} — {close.index[-1].date()}")

    # Preprocesar univariado (Close only) — mismo scaler que v1
    prep   = preprocess(data)
    scaler = prep["scaler"]

    # Obtener array escalado completo para create_sequences_direct
    from src.preprocessor import compute_features, TRAIN_RATIO
    features     = compute_features(data)
    split_idx    = int(len(features) * TRAIN_RATIO)
    train_feat   = features.iloc[:split_idx]
    full_scaled  = scaler.transform(features.values)
    train_scaled = full_scaled[:split_idx]

    for n_steps in [5, 11]:
        print(f"\n{'='*50}")
        print(f"  n_steps = {n_steps}")
        print(f"{'='*50}")

        X_tr, y_tr = create_sequences_direct(train_scaled, n_steps=n_steps)
        # Test: incluye contexto de los ultimos SEQUENCE_LENGTH dias de train
        test_input  = full_scaled[split_idx - SEQUENCE_LENGTH:]
        X_te, y_te  = create_sequences_direct(test_input, n_steps=n_steps)

        print(f"X_train: {X_tr.shape} | y_train: {y_tr.shape}")
        print(f"X_test:  {X_te.shape} | y_test:  {y_te.shape}")

        # LSTM directo
        lstm_model, lstm_hist = train_direct("lstm", X_tr, y_tr, n_steps=n_steps)
        plot_direct_training(lstm_hist, "lstm", n_steps)
        lstm_res = evaluate_direct(lstm_model, X_te, y_te, scaler)

        # GRU directo
        gru_model, gru_hist = train_direct("gru", X_tr, y_tr, n_steps=n_steps)
        plot_direct_training(gru_hist, "gru", n_steps)
        gru_res = evaluate_direct(gru_model, X_te, y_te, scaler)

        print(f"\n  Resumen n_steps={n_steps}:")
        print(f"  Direct LSTM: MAE={lstm_res['mae']:.2f}  RMSE={lstm_res['rmse']:.2f}")
        print(f"  Direct GRU:  MAE={gru_res['mae']:.2f}  RMSE={gru_res['rmse']:.2f}")
        print(f"  (Referencia recursivo: LSTM MAE=126, GRU MAE=110)")

    print("\nEntrenamiento directo completado.")
    print("Modelos guardados: direct_lstm_5d.keras, direct_gru_5d.keras,")
    print("                   direct_lstm_11d.keras, direct_gru_11d.keras")


def train_direct_v4_models():
    """
    Fase 1 SOTA: Direct MIMO v4 con log_return como TARGET.

    Mejoras sobre train_direct_models() (v1):
      - 8 features: log_return(target), close_norm, sma20/50_ratio, rsi14,
        macd_norm, bb_width, atr14_norm
      - TARGET: log_return (estacionario) en lugar de precio absoluto
      - Reconstruccion de precios via exp() en evaluate/predict
    Guarda: direct_lstm_v4_5d.keras, direct_gru_v4_5d.keras
    """
    print("=" * 65)
    print("  IBEX 35 — Fase 1 SOTA: Direct MIMO v4 (log_return target)")
    print("=" * 65)

    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos: {len(close)} dias | {close.index[0].date()} — {close.index[-1].date()}")

    prep        = preprocess_v4(data)
    scaler      = prep["scaler"]
    full_scaled = prep["full_scaled"]
    split_idx   = prep["split_idx"]
    close_real  = prep["close_real"]
    features_v4 = prep["features"]

    n_steps = 5
    print(f"\n{'='*50}")
    print(f"  n_steps={n_steps} | features=8 | target=log_return")
    print(f"{'='*50}")

    # Secuencias de entrenamiento
    train_scaled = full_scaled[:split_idx]
    close_train  = close_real[:split_idx]
    X_tr, y_tr, anchors_tr = create_sequences_direct_v4(
        train_scaled, close_train, n_steps=n_steps)

    # Secuencias de test con contexto de los ultimos seq_length dias de train
    test_ctx   = full_scaled[split_idx - SEQUENCE_LENGTH:]
    close_ctx  = close_real[split_idx - SEQUENCE_LENGTH:]
    X_te, y_te, anchors_te = create_sequences_direct_v4(
        test_ctx, close_ctx, n_steps=n_steps)

    print(f"X_train: {X_tr.shape} | y_train: {y_tr.shape}")
    print(f"X_test:  {X_te.shape} | y_test:  {y_te.shape}")

    # LSTM v4
    lstm_model, lstm_hist = train_direct_v4("lstm", X_tr, y_tr, n_steps=n_steps)
    plot_direct_training(lstm_hist, "lstm_v4", n_steps)
    lstm_res = evaluate_direct_v4(lstm_model, X_te, y_te, anchors_te, scaler)

    # GRU v4
    gru_model, gru_hist = train_direct_v4("gru", X_tr, y_tr, n_steps=n_steps)
    plot_direct_training(gru_hist, "gru_v4", n_steps)
    gru_res = evaluate_direct_v4(gru_model, X_te, y_te, anchors_te, scaler)

    # Tabla comparativa
    print(f"\n{'='*65}")
    print("  COMPARATIVA: v4 (log_return) vs v1 (precio absoluto)")
    print(f"{'='*65}")
    print(f"  {'Modelo':<32} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Direct LSTM v1 (Close, 5d)':<32} {'162':>8} {'--':>8}  (referencia)")
    print(f"  {'Direct GRU  v1 (Close, 5d)':<32} {'165':>8} {'--':>8}  (referencia)")
    print(f"  {'Direct LSTM v4 (log_ret, 5d)':<32} {lstm_res['mae']:>8.2f} {lstm_res['rmse']:>8.2f}  <- NUEVO")
    print(f"  {'Direct GRU  v4 (log_ret, 5d)':<32} {gru_res['mae']:>8.2f} {gru_res['rmse']:>8.2f}  <- NUEVO")
    print(f"{'='*65}")

    for nombre, res in [("LSTM v4", lstm_res), ("GRU v4", gru_res)]:
        steps_str = "  ".join(f"D{s+1}:{res['mae_per_step'][s]:.0f}" for s in range(n_steps))
        print(f"  {nombre} MAE por paso: {steps_str}")

    # Prediccion a 5 dias con los datos mas recientes
    last_60_raw = features_v4.values[-SEQUENCE_LENGTH:]
    last_60_sc  = scaler.transform(last_60_raw)
    anchor      = float(close.iloc[-1])

    lstm_5d = predict_direct_v4(lstm_model, last_60_sc, scaler, anchor)
    gru_5d  = predict_direct_v4(gru_model,  last_60_sc, scaler, anchor)

    last_date = close.index[-1]
    dates_5d  = pd.bdate_range(start=last_date, periods=6)[1:]

    print(f"\n  Ultimo precio real: {anchor:.2f} pts ({last_date.date()})")
    print(f"\n  {'Fecha':<14} {'LSTM v4':>10} {'GRU v4':>10}   {'Var LSTM':>10} {'Var GRU':>10}")
    print(f"  {'-'*60}")
    for d, lp, gp in zip(dates_5d, lstm_5d, gru_5d):
        dl, dg = lp - anchor, gp - anchor
        sl, sg = ("+" if dl >= 0 else ""), ("+" if dg >= 0 else "")
        print(f"  {str(d.date()):<14} {lp:>10.2f} {gp:>10.2f}   {sl}{dl:>9.2f} {sg}{dg:>9.2f}")

    print("\nEntrenamiento v4 completado.")
    print("Modelos: direct_lstm_v4_5d.keras, direct_gru_v4_5d.keras")


def train_attention_models():
    """
    Fase 2 SOTA: LSTM + Multi-Head Self-Attention con pipeline v4.

    Usa exactamente el mismo preprocesamiento que train_direct_v4_models():
      - 8 features v4 (log_return target, close_norm, indicadores tecnicos)
      - scaler_v4.pkl
      - create_sequences_direct_v4 + evaluate_direct_v4

    Diferencia arquitectonica vs v4:
      v4:   Input -> LSTM(100, rs=False) -> Dense(50) -> Dense(5)
      Attn: Input -> LSTM(128, rs=True) -> MultiHeadAttention(4h) ->
            GlobalAvgPool -> Dense(64,gelu) -> Dense(5)

    Guarda: attention_lstm_5d.keras
    Objetivo: bajar de MAE=142 (GRU v4) hacia ~120-130.
    """
    print("=" * 65)
    print("  IBEX 35 — Fase 2 SOTA: LSTM + Multi-Head Attention")
    print("=" * 65)

    data  = load_ibex35(full_history=True)
    close = data["Close"]
    print(f"Datos: {len(close)} dias | {close.index[0].date()} — {close.index[-1].date()}")

    # Reutiliza pipeline v4 exacto
    prep        = preprocess_v4(data)
    scaler      = prep["scaler"]
    full_scaled = prep["full_scaled"]
    split_idx   = prep["split_idx"]
    close_real  = prep["close_real"]
    features_v4 = prep["features"]

    n_steps = 5

    # Secuencias de entrenamiento
    train_scaled = full_scaled[:split_idx]
    close_train  = close_real[:split_idx]
    X_tr, y_tr, anchors_tr = create_sequences_direct_v4(
        train_scaled, close_train, n_steps=n_steps)

    # Secuencias de test con contexto
    test_ctx  = full_scaled[split_idx - SEQUENCE_LENGTH:]
    close_ctx = close_real[split_idx - SEQUENCE_LENGTH:]
    X_te, y_te, anchors_te = create_sequences_direct_v4(
        test_ctx, close_ctx, n_steps=n_steps)

    print(f"X_train: {X_tr.shape} | y_train: {y_tr.shape}")
    print(f"X_test:  {X_te.shape} | y_test:  {y_te.shape}")

    # Entrenar Attention LSTM
    attn_model, attn_hist = train_attention(X_tr, y_tr, n_steps=n_steps)
    plot_attention_training(attn_hist, n_steps)
    attn_res = evaluate_direct_v4(attn_model, X_te, y_te, anchors_te, scaler)

    # Tabla comparativa completa
    print(f"\n{'='*65}")
    print("  COMPARATIVA: Attention vs v4 vs v1")
    print(f"{'='*65}")
    print(f"  {'Modelo':<35} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*53}")
    print(f"  {'Direct LSTM v1 (Close, 5d)':<35} {'162':>8} {'--':>8}  (ref)")
    print(f"  {'Direct GRU  v1 (Close, 5d)':<35} {'165':>8} {'--':>8}  (ref)")
    print(f"  {'Direct LSTM v4 (log_ret, 5d)':<35} {'144':>8} {'206':>8}  (ref)")
    print(f"  {'Direct GRU  v4 (log_ret, 5d)':<35} {'142':>8} {'205':>8}  (ref)")
    print(f"  {'LSTM + Attention v4 (log_ret)':<35} {attn_res['mae']:>8.2f} {attn_res['rmse']:>8.2f}  <- NUEVO")
    print(f"{'='*65}")

    steps_str = "  ".join(f"D{s+1}:{attn_res['mae_per_step'][s]:.0f}" for s in range(n_steps))
    print(f"\n  Attention MAE por paso: {steps_str}")

    # Mejora vs mejor v4
    best_v4_mae = 141.79
    if attn_res["mae"] < best_v4_mae:
        imp = (best_v4_mae - attn_res["mae"]) / best_v4_mae * 100
        print(f"\n  RESULTADO: Attention MEJORA al GRU v4 en {imp:.1f}%")
    else:
        diff = attn_res["mae"] - best_v4_mae
        print(f"\n  RESULTADO: Attention NO supera a GRU v4 (diferencia: +{diff:.2f} MAE)")
        print(f"  -> GRU v4 sigue siendo el mejor modelo.")

    # Prediccion a 5 dias
    last_60_raw = features_v4.values[-SEQUENCE_LENGTH:]
    last_60_sc  = scaler.transform(last_60_raw)
    anchor      = float(close.iloc[-1])
    attn_5d     = predict_attention(attn_model, last_60_sc, scaler, anchor)

    last_date = close.index[-1]
    dates_5d  = pd.bdate_range(start=last_date, periods=6)[1:]

    print(f"\n  Ultimo precio real: {anchor:.2f} pts ({last_date.date()})")
    print(f"\n  {'Fecha':<14} {'Attention':>12}   {'Variacion':>10}")
    print(f"  {'-'*42}")
    for d, ap in zip(dates_5d, attn_5d):
        da = ap - anchor
        print(f"  {str(d.date()):<14} {ap:>12.2f}   {('+' if da>=0 else '')}{da:>9.2f}")

    print("\nFase 2 completada. Modelo: attention_lstm_5d.keras")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "hybrid":
        train_hybrid_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "direct":
        train_direct_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "v4":
        train_direct_v4_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "attention":
        train_attention_models()
    else:
        train_v3()
