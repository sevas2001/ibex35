import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR  = Path(__file__).parent.parent / "models"
PLOTS_DIR   = Path(__file__).parent.parent / "data" / "plots"
SEQUENCE_LENGTH = 60
N_FEATURES      = 1   # Close (univariado)


def _inv_close(scaler, arr_1d: np.ndarray) -> np.ndarray:
    """
    Inverse-transform un array 1D de valores Close escalados.
    """
    dummy = np.zeros((len(arr_1d), N_FEATURES))
    dummy[:, 0] = arr_1d
    return scaler.inverse_transform(dummy)[:, 0]


def build_gru(seq_length: int = SEQUENCE_LENGTH,
              n_features: int = N_FEATURES) -> object:
    """
    Construye la arquitectura GRU.
    GRU (Cho et al., 2014): 2 puertas (update + reset) vs 3 del LSTM.
    Mismos hiperparametros que el LSTM para comparacion justa.
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
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    return model


def train_gru(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> tuple:
    """
    Entrena el modelo GRU con EarlyStopping.
    Retorna (model, history).
    """
    from tensorflow import keras

    n_features = X_train.shape[2]
    model = build_gru(n_features=n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    validation_data = (X_val, y_val) if X_val is not None else None

    print("Entrenando GRU...")
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
    model.save(MODELS_DIR / "gru_model.keras")
    print("Modelo GRU guardado en models/gru_model.keras")
    return model, history


def plot_gru_training_history(history) -> None:
    """Grafica la curva de loss durante el entrenamiento del GRU."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("GRU -- Curva de Aprendizaje")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = PLOTS_DIR / "gru_training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Curva de aprendizaje GRU guardada: {path}")


def evaluate_gru(model, X_test: np.ndarray, y_test: np.ndarray,
                 scaler) -> dict:
    """Genera predicciones y calcula metricas en escala real."""
    predictions_scaled = model.predict(X_test).flatten()

    predictions = _inv_close(scaler, predictions_scaled)
    actuals     = _inv_close(scaler, y_test)

    mse  = np.mean((actuals - predictions) ** 2)
    mae  = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(mse)

    print(f"\nGRU -- Metricas en test:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    return {"predictions": predictions, "actuals": actuals,
            "mse": mse, "mae": mae, "rmse": rmse}


def predict_next_5_days_gru(model, last_60_features_scaled: np.ndarray,
                             scaler) -> list:
    """
    Prediccion autoregresiva a 5 dias con modelo GRU univariado.
    """
    input_seq = last_60_features_scaled.tolist()

    predictions_scaled = []
    for _ in range(5):
        X = np.array(input_seq[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        pred_scaled = float(model.predict(X, verbose=0)[0, 0])
        predictions_scaled.append(pred_scaled)
        input_seq.append([pred_scaled])

    return _inv_close(scaler, np.array(predictions_scaled)).tolist()
