"""
attention_model.py — LSTM + Multi-Head Self-Attention (Fase 2 SOTA)

Problema que resuelve:
  El LSTM estandar comprime toda la ventana de 60 dias en un unico
  vector de estado h_t. Dias relevantes (rebotes, crashes recientes)
  tienen el mismo peso que dias irrelevantes.

Solucion — Self-Attention sobre la secuencia LSTM:
  Input(60) -> LSTM(128, return_sequences=True) -> 60 vectores h_1..h_60
                                                 |
                                       MultiHeadAttention(4 heads)
                                         pondera que dias importan
                                                 |
                                       GlobalAvgPooling -> Dense(n_steps)

Ref: Vaswani et al. (2017) "Attention is All You Need"
     Qin et al. (2017)    "A Dual-Stage Attention-Based RNN for Time Series"
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR      = Path(__file__).parent.parent / "models"
PLOTS_DIR       = Path(__file__).parent.parent / "data" / "plots"
SEQUENCE_LENGTH = 60


def build_lstm_attention(seq_length: int = SEQUENCE_LENGTH,
                          n_features: int = 8,
                          n_steps: int = 5):
    """
    LSTM(128, return_sequences=True) + MultiHeadAttention(4 heads) + Dense(n_steps).

    Arquitectura:
      Input(seq_length, n_features)
        -> LSTM(128, rs=True)          # (batch, 60, 128) — todos los estados
        -> Dropout(0.2)
        -> MultiHeadAttention(4, 32)   # self-attention: cada dia ve a los demas
        -> LayerNorm + residual
        -> GlobalAveragePooling1D      # (batch, 128) — promedio ponderado
        -> Dense(64, gelu)
        -> Dropout(0.2)
        -> Dense(n_steps)              # MIMO: n_steps salidas simultaneas

    key_dim=32: dimension de las proyecciones Q/K/V por cabeza.
    gelu: activacion canonica para Transformers (mejor que relu para gradientes suaves).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(seq_length, n_features))

    # LSTM — captura dependencias temporales, devuelve toda la secuencia
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)

    # Multi-Head Self-Attention — cada posicion atiende a todas las demas
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)   # residual + norm

    # Colapsar la secuencia en un vector resumen
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_steps)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


def train_attention(X_train: np.ndarray,
                    y_train: np.ndarray,
                    n_steps: int = 5) -> tuple:
    """
    Entrena LSTM + Attention con target log_return (v4).
    Usa mismos callbacks que los modelos directos v4.
    Guarda como attention_lstm_{n_steps}d.keras.
    """
    from tensorflow import keras

    n_features = X_train.shape[2]
    model      = build_lstm_attention(n_features=n_features, n_steps=n_steps)
    save_name  = f"attention_lstm_{n_steps}d.keras"

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

    print(f"Entrenando LSTM + Attention (n_steps={n_steps}, n_features={n_features})...")
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
    print(f"Modelo guardado: models/{save_name}")
    return model, history


def predict_attention(model,
                       last_60_scaled: np.ndarray,
                       scaler,
                       anchor_price: float) -> list:
    """
    Prediccion con LSTM+Attention — misma interfaz que predict_direct_v4().

    last_60_scaled: (60, n_features) — features v4 escaladas
    anchor_price:   ultimo precio de cierre conocido
    Retorna lista de n_steps precios reales.
    """
    from src.direct_model import reconstruct_prices_from_returns

    X            = last_60_scaled.reshape(1, 60, last_60_scaled.shape[1])
    preds_scaled = model(X, training=False).numpy()[0]   # (n_steps,) — evita TF retrace
    return reconstruct_prices_from_returns(preds_scaled, anchor_price, scaler).tolist()


def plot_attention_training(history, n_steps: int) -> None:
    """Curva de aprendizaje del modelo Attention."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"LSTM + Attention MIMO ({n_steps} pasos) — Curva de Aprendizaje")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / f"attention_lstm_{n_steps}d_training.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curva guardada: {path}")
