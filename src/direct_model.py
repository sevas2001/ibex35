"""
direct_model.py — Direct Multi-Step (MIMO) Prediction

Soluciona el problema de "prediccion plana" de la estrategia autoregresiva.

Estrategia recursiva (lo que teniamos):
  Input(60) -> LSTM -> Dense(1) -> re-alimentar -> Dense(1) -> ...
  Problema: el error se acumula en cada paso y converge a la media.

Estrategia directa MIMO (Multi-Input Multi-Output):
  Input(60) -> LSTM -> Dense(N)  <- predice N dias en UN solo forward pass
  No hay re-alimentacion, no hay propagacion de error.

Ref: Taieb & Hyndman (2012) "Recursive and direct multi-step forecasting:
     the best of both worlds" — confirma que direct supera a recursive
     en la mayoria de benchmarks de series temporales financieras.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR      = Path(__file__).parent.parent / "models"
PLOTS_DIR       = Path(__file__).parent.parent / "data" / "plots"
SEQUENCE_LENGTH = 60


def create_sequences_direct(features_scaled: np.ndarray,
                             seq_length: int = SEQUENCE_LENGTH,
                             n_steps: int = 5) -> tuple:
    """
    Crea secuencias para prediccion directa multi-step (MIMO).

    X[i]: ventana (seq_length, n_features)  — input
    y[i]: (n_steps,)                        — proximos n_steps valores de Close (col 0)

    A diferencia de create_sequences() estandar, y tiene forma (samples, n_steps)
    en lugar de (samples,). El modelo aprende a predecir todos los pasos a la vez.
    """
    X, y = [], []
    for i in range(seq_length, len(features_scaled) - n_steps + 1):
        X.append(features_scaled[i - seq_length:i])       # (seq_length, n_features)
        y.append(features_scaled[i: i + n_steps, 0])      # (n_steps,) — solo Close
    return np.array(X), np.array(y)


def build_direct_lstm(seq_length: int = SEQUENCE_LENGTH,
                      n_features: int = 1,
                      n_steps: int = 5,
                      loss: str = "mean_squared_error"):
    """
    LSTM con salida directa de n_steps dias.
    loss: "mean_squared_error" (conservador) o "mean_absolute_error" (mas atrevido).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.LSTM(100, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(100, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dense(n_steps),
    ])
    model.compile(optimizer="adam", loss=loss)
    model.summary()
    return model


def build_direct_gru(seq_length: int = SEQUENCE_LENGTH,
                     n_features: int = 1,
                     n_steps: int = 5,
                     loss: str = "mean_squared_error"):
    """
    GRU con salida directa de n_steps dias.
    loss: "mean_squared_error" (conservador) o "mean_absolute_error" (mas atrevido).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.GRU(100, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(100, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dense(n_steps),
    ])
    model.compile(optimizer="adam", loss=loss)
    model.summary()
    return model


def train_direct(model_type: str,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 n_steps: int = 5) -> tuple:
    """
    Entrena modelo directo LSTM o GRU.
    y_train shape: (samples, n_steps)  <-- multi-output target
    Guarda como direct_lstm_{n_steps}d.keras o direct_gru_{n_steps}d.keras
    """
    from tensorflow import keras

    n_features = X_train.shape[2]
    if model_type == "lstm":
        model = build_direct_lstm(n_features=n_features, n_steps=n_steps)
        save_name = f"direct_lstm_{n_steps}d.keras"
    else:
        model = build_direct_gru(n_features=n_features, n_steps=n_steps)
        save_name = f"direct_gru_{n_steps}d.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            min_delta=1e-5,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Entrenando Direct {model_type.upper()} (n_steps={n_steps})...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / save_name)
    print(f"Modelo directo guardado: models/{save_name}")
    return model, history


def evaluate_direct(model, X_test: np.ndarray, y_test_scaled: np.ndarray,
                    scaler) -> dict:
    """
    Evalua modelo directo en test set.
    y_test_scaled shape: (samples, n_steps)
    Retorna MAE/RMSE promedio sobre todos los pasos y por paso individual.
    """
    preds_scaled  = model.predict(X_test)        # (samples, n_steps)
    n_steps       = preds_scaled.shape[1]
    nf            = scaler.n_features_in_

    preds_real   = np.zeros_like(preds_scaled)
    actuals_real = np.zeros_like(y_test_scaled)

    for s in range(n_steps):
        dummy = np.zeros((len(preds_scaled), nf))
        dummy[:, 0] = preds_scaled[:, s]
        preds_real[:, s] = scaler.inverse_transform(dummy)[:, 0]

        dummy[:, 0] = y_test_scaled[:, s]
        actuals_real[:, s] = scaler.inverse_transform(dummy)[:, 0]

    mae_per_step  = np.mean(np.abs(actuals_real - preds_real), axis=0)
    rmse_per_step = np.sqrt(np.mean((actuals_real - preds_real) ** 2, axis=0))
    mae_global    = float(np.mean(mae_per_step))
    rmse_global   = float(np.mean(rmse_per_step))

    print(f"\nDirect MIMO — Metricas en test (promedio {n_steps} pasos):")
    print(f"  MAE global:  {mae_global:.2f}")
    print(f"  RMSE global: {rmse_global:.2f}")
    for s in range(n_steps):
        print(f"  Paso {s+1}: MAE={mae_per_step[s]:.2f}  RMSE={rmse_per_step[s]:.2f}")

    return {
        "preds_real":    preds_real,
        "actuals_real":  actuals_real,
        "mae_per_step":  mae_per_step,
        "rmse_per_step": rmse_per_step,
        "mae":           mae_global,
        "rmse":          rmse_global,
    }


def predict_direct(model, last_60_scaled: np.ndarray, scaler) -> list:
    """
    Prediccion directa: UN solo forward pass genera todos los dias.
    No hay re-alimentacion — sin propagacion de error.
    last_60_scaled: (60, n_features)
    Retorna lista de precios en escala real (longitud = n_steps del modelo).
    """
    nf = scaler.n_features_in_
    X  = last_60_scaled.reshape(1, 60, last_60_scaled.shape[1])
    preds_scaled = model.predict(X, verbose=0)[0]   # (n_steps,)

    dummy = np.zeros((len(preds_scaled), nf))
    dummy[:, 0] = preds_scaled
    return scaler.inverse_transform(dummy)[:, 0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# v4 — Log Return Target (Fase 1 SOTA)
# El modelo predice log_returns (serie estacionaria) en lugar de precios absolutos.
# Los precios reales se reconstruyen via reconstruct_prices_from_returns().
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences_direct_v4(features_scaled: np.ndarray,
                                close_real: np.ndarray,
                                seq_length: int = SEQUENCE_LENGTH,
                                n_steps: int = 5) -> tuple:
    """
    Secuencias para prediccion directa v4 (log_return como target).

    X[i]:       (seq_length, n_features) — ventana de contexto escalada
    y[i]:       (n_steps,)  — log_returns escalados de los proximos n_steps dias
    anchors[i]: float       — precio real en t-1 (para reconstruir precios)

    El anchor es el ultimo precio conocido ANTES de la ventana de prediccion.
    Se necesita para reconstruct_prices_from_returns() en evaluacion y produccion.
    """
    X, y, anchors = [], [], []
    for i in range(seq_length, len(features_scaled) - n_steps + 1):
        X.append(features_scaled[i - seq_length:i])   # (seq_length, n_features)
        y.append(features_scaled[i: i + n_steps, 0])  # (n_steps,) scaled log_returns
        anchors.append(close_real[i - 1])              # precio real en t-1
    return np.array(X), np.array(y), np.array(anchors)


def reconstruct_prices_from_returns(log_returns_scaled: np.ndarray,
                                     anchor_prices,
                                     scaler) -> np.ndarray:
    """
    Convierte log_returns escalados a precios reales via exp().

    Proceso:
      1. Inverse-transform col 0 del scaler -> retornos logaritmicos reales
      2. price[s] = price[s-1] * exp(r[s]),  price[-1] = anchor_price

    Args:
      log_returns_scaled: (samples, n_steps) o (n_steps,) para prediccion individual
      anchor_prices:      (samples,) o float para prediccion individual
      scaler:             MinMaxScaler ajustado sobre features v4 (col 0 = log_return)

    Returns:
      prices: (samples, n_steps) o (n_steps,) segun el input
    """
    nf     = scaler.n_features_in_
    single = log_returns_scaled.ndim == 1
    if single:
        log_returns_scaled = log_returns_scaled[np.newaxis, :]
        anchor_prices      = np.array([anchor_prices], dtype=float)

    n, n_steps = log_returns_scaled.shape
    prices     = np.zeros((n, n_steps))

    for i in range(n):
        dummy       = np.zeros((n_steps, nf))
        dummy[:, 0] = log_returns_scaled[i]
        r_actual    = scaler.inverse_transform(dummy)[:, 0]
        p = float(anchor_prices[i])
        for s, r in enumerate(r_actual):
            p = p * np.exp(r)
            prices[i, s] = p

    return prices[0] if single else prices


def evaluate_direct_v4(model,
                        X_test: np.ndarray,
                        y_test_scaled: np.ndarray,
                        anchor_prices: np.ndarray,
                        scaler) -> dict:
    """
    Evalua modelo directo v4 en test set.
    Reconstruye precios reales a partir de log_returns para MAE/RMSE en puntos IBEX.

    y_test_scaled shape: (samples, n_steps) — log_returns escalados
    anchor_prices shape: (samples,)         — precios reales en t-1
    """
    preds_scaled = model.predict(X_test)          # (samples, n_steps)
    n_steps      = preds_scaled.shape[1]

    preds_real   = reconstruct_prices_from_returns(preds_scaled,  anchor_prices, scaler)
    actuals_real = reconstruct_prices_from_returns(y_test_scaled, anchor_prices, scaler)

    mae_per_step  = np.mean(np.abs(actuals_real - preds_real), axis=0)
    rmse_per_step = np.sqrt(np.mean((actuals_real - preds_real) ** 2, axis=0))
    mae_global    = float(np.mean(mae_per_step))
    rmse_global   = float(np.mean(rmse_per_step))

    print(f"\nDirect v4 MIMO (log_return target) — Metricas en test ({n_steps} pasos):")
    print(f"  MAE global:  {mae_global:.2f}")
    print(f"  RMSE global: {rmse_global:.2f}")
    for s in range(n_steps):
        print(f"  Paso {s+1}: MAE={mae_per_step[s]:.2f}  RMSE={rmse_per_step[s]:.2f}")

    return {
        "preds_real":    preds_real,
        "actuals_real":  actuals_real,
        "mae_per_step":  mae_per_step,
        "rmse_per_step": rmse_per_step,
        "mae":           mae_global,
        "rmse":          rmse_global,
    }


def predict_direct_v4(model,
                       last_60_scaled: np.ndarray,
                       scaler,
                       anchor_price: float) -> list:
    """
    Prediccion directa v4: un solo forward pass -> log_returns -> precios reales.

    Pasos:
      1. forward pass -> (n_steps,) log_returns escalados
      2. inverse_transform col 0 -> log_returns reales
      3. price[t] = anchor_price * exp(cumsum(r[0:t+1]))

    last_60_scaled: (60, n_features) — features v4 escaladas
    anchor_price:   ultimo precio de cierre conocido (close[-1])
    Retorna lista de n_steps precios predichos.
    """
    X            = last_60_scaled.reshape(1, 60, last_60_scaled.shape[1])
    preds_scaled = model.predict(X, verbose=0)[0]   # (n_steps,)
    return reconstruct_prices_from_returns(preds_scaled, anchor_price, scaler).tolist()


def train_direct_v4(model_type: str,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     n_steps: int = 5) -> tuple:
    """
    Entrena modelo directo v4 (target = log_return).
    Reutiliza las mismas arquitecturas LSTM/GRU de train_direct() —
    solo cambia el target semantico (log_return vs precio absoluto).
    Guarda como direct_lstm_v4_{n}d.keras o direct_gru_v4_{n}d.keras.
    """
    from tensorflow import keras

    n_features = X_train.shape[2]
    if model_type == "lstm":
        model     = build_direct_lstm(n_features=n_features, n_steps=n_steps)
        save_name = f"direct_lstm_v4_{n_steps}d.keras"
    else:
        model     = build_direct_gru(n_features=n_features, n_steps=n_steps)
        save_name = f"direct_gru_v4_{n_steps}d.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
            min_delta=1e-5,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Entrenando Direct v4 {model_type.upper()} "
          f"(n_steps={n_steps}, target=log_return, n_features={n_features})...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / save_name)
    print(f"Modelo v4 guardado: models/{save_name}")
    return model, history


def plot_direct_training(history, model_type: str, n_steps: int) -> None:
    """Curva de aprendizaje del modelo directo."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Direct {model_type.upper()} MIMO ({n_steps} pasos) — Curva de Aprendizaje")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / f"direct_{model_type}_{n_steps}d_training.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curva guardada: {path}")
