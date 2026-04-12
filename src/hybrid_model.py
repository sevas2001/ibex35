"""
hybrid_model.py — Modelo híbrido ARIMA + LSTM

Estrategia:
  1. ARIMA(5,1,0) modela la componente lineal/tendencia del precio.
  2. LSTM aprende el residuo no lineal: residuo_t = real_t - arima_pred_t
  3. Prediccion final: hybrid_t = arima_pred_t + lstm_residual_t

Ventajas sobre LSTM puro:
  - El residuo es estacionario (media ~0) → LSTM aprende mejor
  - ARIMA maneja la tendencia sin error proporcional (β₁=1.000)
  - Las features v2 (log_return, RSI, MA50) mejoran la directional accuracy
    del componente residual
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from sklearn.preprocessing import MinMaxScaler

MODELS_DIR  = Path(__file__).parent.parent / "models"
PLOTS_DIR   = Path(__file__).parent.parent / "data" / "plots"
SEQ_LEN     = 60
N_FEAT_HYB  = 4    # residuo + log_return + ma50_ratio + rsi14


# ── Preparacion de datos para el hibrido ─────────────────────────────────────

def build_arima_residuals(close: pd.Series,
                          train_size: int,
                          arima_order: tuple = (5, 1, 0)) -> tuple:
    """
    Genera predicciones ARIMA one-step-ahead sobre toda la serie
    usando expanding window en train y rolling en test.

    Retorna:
      arima_preds  : np.ndarray (len(close),)  — predicciones ARIMA alineadas
      residuals    : np.ndarray (len(close),)  — real - arima_pred
    """
    vals = close.values.astype(float)
    arima_preds = np.full(len(vals), np.nan)

    print(f"  Generando residuos ARIMA sobre {len(vals)} dias...")
    for t in range(max(arima_order[0] + 1, 30), len(vals)):
        window = vals[:t]
        try:
            fit = ARIMAModel(window, order=arima_order).fit()
            arima_preds[t] = float(fit.forecast(steps=1).iloc[0])
        except Exception:
            arima_preds[t] = vals[t - 1]   # fallback: naive

        if t % 1000 == 0:
            print(f"    {t}/{len(vals)} dias...")

    residuals = vals - arima_preds
    print(f"  Residuos ARIMA: media={np.nanmean(residuals):.2f}  std={np.nanstd(residuals):.2f}")
    return arima_preds, residuals


def prepare_hybrid_sequences(residuals: np.ndarray,
                              features_v2: np.ndarray,
                              train_size: int) -> dict:
    """
    Construye secuencias (X, y) para el LSTM del hibrido.

    X[i]: (SEQ_LEN, N_FEAT_HYB) — ventana de features v2 + residuo
    y[i]: residuo del dia siguiente (target)

    features_v2 ya incluye close en col 0; sustituimos col 0 por residuo
    para que el LSTM aprenda directamente la componente no lineal.
    """
    # Combinar: [residuo, log_return, ma50_ratio, rsi14]
    n = len(residuals)
    combined = np.column_stack([
        residuals,               # col 0: residuo (target)
        features_v2[:, 1],       # col 1: log_return
        features_v2[:, 2],       # col 2: ma50_ratio
        features_v2[:, 3],       # col 3: rsi14
    ])

    # Scaler sobre train (sin leakage)
    scaler_hyb = MinMaxScaler(feature_range=(-1, 1))
    scaler_hyb.fit(combined[:train_size])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_hyb, MODELS_DIR / "scaler_hybrid.pkl")

    scaled = scaler_hyb.transform(combined)

    # Indice de inicio: primer punto con residuo valido + SEQ_LEN
    start = np.argmax(~np.isnan(residuals)) + SEQ_LEN
    # Asegurar que start esta dentro de train/test
    start = max(start, SEQ_LEN)

    X, y = [], []
    for i in range(start, n):
        X.append(scaled[i - SEQ_LEN:i])   # (60, 4)
        y.append(scaled[i, 0])             # residuo escalado
    X = np.array(X)
    y = np.array(y)

    # Split manteniendo proporcion temporal
    split = train_size - SEQ_LEN - (start - SEQ_LEN)
    split = max(0, min(split, len(X) - 1))

    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test":  X[split:],
        "y_test":  y[split:],
        "scaler":  scaler_hyb,
        "combined": combined,
        "scaled":   scaled,
        "start_idx": start,
        "split_idx": split,
    }


# ── Arquitectura LSTM del hibrido ─────────────────────────────────────────────

def build_hybrid_lstm(seq_len: int = SEQ_LEN,
                      n_features: int = N_FEAT_HYB):
    """LSTM con Attention manual para el componente residual."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(seq_len, n_features))

    # LSTM bidireccional — captura contexto pasado y futuro en train
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)

    # Attention: pondera los pasos temporales mas relevantes
    attn = layers.Dense(1, activation="tanh")(x)          # (batch, seq, 1)
    attn = layers.Flatten()(attn)                          # (batch, seq)
    attn = layers.Activation("softmax")(attn)              # pesos normalizados
    attn = layers.RepeatVector(64 * 2)(attn)               # (batch, 128, seq)
    attn = layers.Permute([2, 1])(attn)                    # (batch, seq, 128)
    x = layers.Multiply()([x, attn])                       # apply attention
    x = layers.Lambda(lambda z: z[:, -1, :])(x)           # ultimo paso

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1)(x)

    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",    # Huber loss: menos sensible a outliers que MSE
    )
    model.summary()
    return model


def train_hybrid_lstm(X_train, y_train):
    """Entrena el LSTM residual con EarlyStopping + ReduceLR."""
    from tensorflow import keras

    model = build_hybrid_lstm(n_features=X_train.shape[2])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7
        ),
    ]

    print("Entrenando LSTM residual (hibrido)...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODELS_DIR / "hybrid_lstm.keras")
    print("Modelo hibrido guardado: models/hybrid_lstm.keras")
    return model, history


# ── Evaluacion ────────────────────────────────────────────────────────────────

def evaluate_hybrid(hybrid_model, prep: dict,
                    arima_preds: np.ndarray,
                    close_vals: np.ndarray,
                    train_size: int) -> dict:
    """
    Evalua el hibrido en el test set.
    hybrid_pred = arima_pred + lstm_residual_pred (en escala real)
    """
    X_test  = prep["X_test"]
    scaler  = prep["scaler"]
    start   = prep["start_idx"]
    split   = prep["split_idx"]

    # Predicciones de residuo escalado
    resid_scaled_pred = hybrid_model.predict(X_test).flatten()

    # Inverse transform solo col 0 (residuo)
    dummy = np.zeros((len(resid_scaled_pred), N_FEAT_HYB))
    dummy[:, 0] = resid_scaled_pred
    resid_pred = scaler.inverse_transform(dummy)[:, 0]

    # Alinear con el test set real
    test_start_abs = start + split   # indice absoluto en close_vals
    n_test = len(resid_pred)

    actuals_abs   = close_vals[test_start_abs: test_start_abs + n_test]
    arima_abs     = arima_preds[test_start_abs: test_start_abs + n_test]

    hybrid_pred = arima_abs + resid_pred

    # Recortar si hay desajuste de longitud
    min_len = min(len(actuals_abs), len(hybrid_pred))
    actuals_abs  = actuals_abs[:min_len]
    hybrid_pred  = hybrid_pred[:min_len]
    arima_abs    = arima_abs[:min_len]

    mae  = np.mean(np.abs(actuals_abs - hybrid_pred))
    rmse = np.sqrt(np.mean((actuals_abs - hybrid_pred) ** 2))
    mse  = rmse ** 2
    mape = np.mean(np.abs((actuals_abs - hybrid_pred) / actuals_abs)) * 100

    print(f"\nHibrido ARIMA+LSTM — Metricas en test:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  (Referencia ARIMA solo: MAE~85, LSTM solo: MAE~134)")

    return {
        "predictions": hybrid_pred,
        "actuals":     actuals_abs,
        "arima_component": arima_abs,
        "resid_component": resid_pred[:min_len],
        "mse": mse, "mae": mae, "rmse": rmse, "mape": mape,
    }


def plot_hybrid_training(history) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0a0a0f")
    ax.set_facecolor("#111118")
    ax.plot(history.history["loss"],     color="#3b82f6", lw=1.2, label="Train (Huber)")
    ax.plot(history.history["val_loss"], color="#00d4aa", lw=1.2, label="Validation")
    ax.set_title("Hibrido ARIMA+LSTM — Curva de aprendizaje (residuos)",
                 color="#f1f5f9", fontsize=11)
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("Huber Loss", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#1e1e2e", lw=0.4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(facecolor="#16161f", labelcolor="#f1f5f9", edgecolor="#333")
    plt.tight_layout()
    path = PLOTS_DIR / "hybrid_training_history.png"
    plt.savefig(path, dpi=150, facecolor="#0a0a0f")
    plt.close()
    print(f"Curva de aprendizaje hibrido guardada: {path}")


def plot_hybrid_vs_arima(results: dict, test_index: pd.DatetimeIndex) -> None:
    """Grafica: real vs ARIMA solo vs Hibrido en el test set."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n = min(len(results["actuals"]), len(test_index))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                    facecolor="#0a0a0f",
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    for ax in (ax1, ax2):
        ax.set_facecolor("#111118")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(color="#1e1e2e", lw=0.4)
        for sp in ax.spines.values(): sp.set_visible(False)

    idx = test_index[:n]
    ax1.plot(idx, results["actuals"][:n],     color="#e2e8f0", lw=1.0, label="Real")
    ax1.plot(idx, results["arima_component"][:n], color="#fb923c", lw=0.8,
             alpha=0.7, label="ARIMA solo")
    ax1.plot(idx, results["predictions"][:n], color="#00d4aa", lw=1.0,
             alpha=0.9, label=f"Hibrido ARIMA+LSTM (MAE={results['mae']:.1f})")
    ax1.set_title("IBEX 35 — Hibrido ARIMA+LSTM vs ARIMA solo (test set)",
                  color="#f1f5f9", fontsize=12)
    ax1.set_ylabel("Puntos", color="#94a3b8")
    ax1.xaxis.set_visible(False)
    ax1.legend(facecolor="#16161f", labelcolor="#f1f5f9", edgecolor="#333", fontsize=9)

    err_hyb  = np.abs(results["actuals"][:n] - results["predictions"][:n])
    err_arima = np.abs(results["actuals"][:n] - results["arima_component"][:n])
    ax2.fill_between(idx, err_hyb,  alpha=0.5, color="#00d4aa", label=f"Hibrido MAE={results['mae']:.1f}")
    ax2.fill_between(idx, err_arima, alpha=0.4, color="#fb923c", label=f"ARIMA MAE={np.mean(err_arima):.1f}")
    ax2.set_ylabel("|Error| pts", color="#94a3b8")
    ax2.legend(facecolor="#16161f", labelcolor="#f1f5f9", edgecolor="#333", fontsize=8)

    plt.savefig(PLOTS_DIR / "hybrid_vs_arima.png", dpi=150,
                bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    print(f"Grafico hibrido guardado: data/plots/hybrid_vs_arima.png")
