import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"
SEQUENCE_LENGTH = 60


def build_lstm(seq_length: int = SEQUENCE_LENGTH) -> object:
    """Construye la arquitectura LSTM."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        layers.LSTM(100, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(100, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray = None, y_val: np.ndarray = None) -> tuple:
    """
    Entrena el modelo LSTM con EarlyStopping para evitar overfitting.
    Retorna (model, history).
    """
    from tensorflow import keras

    model = build_lstm()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    validation_data = None
    if X_val is not None:
        validation_data = (X_val, y_val)

    print("Entrenando LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1 if validation_data is None else 0.0,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_model.keras")
    print("Modelo LSTM guardado en models/lstm_model.keras")
    return model, history


def plot_training_history(history) -> None:
    """Grafica la curva de loss durante el entrenamiento."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM — Curva de Aprendizaje")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "lstm_training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curva de aprendizaje guardada: {path}")


def evaluate_lstm(model, X_test: np.ndarray, y_test: np.ndarray,
                  scaler) -> dict:
    """Genera predicciones y calcula métricas en escala real (puntos del índice)."""
    predictions_scaled = model.predict(X_test)

    # Invertir normalización para obtener valores reales
    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = np.mean((actuals - predictions) ** 2)
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(mse)

    print(f"\nLSTM — Metricas en test:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    return {"predictions": predictions, "actuals": actuals,
            "mse": mse, "mae": mae, "rmse": rmse}


def predict_next_5_days(model, last_60_days_scaled: np.ndarray, scaler) -> list:
    """
    Predicción autoregresiva a 5 días.
    Cada predicción se usa como input del día siguiente.
    """
    input_seq = last_60_days_scaled.copy().tolist()
    predictions = []

    for _ in range(5):
        X = np.array(input_seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        pred_scaled = model.predict(X, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        input_seq.append([pred_scaled])

    # Invertir normalización
    predictions_real = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    return predictions_real.tolist()
