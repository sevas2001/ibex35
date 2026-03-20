import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
MODELS_DIR = Path(__file__).parent.parent / "models"

SEQUENCE_LENGTH = 60  # días de historia como input
TRAIN_RATIO = 0.80


def split_data(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split temporal 80/20 sin shuffle para respetar el orden cronológico."""
    close = data["Close"]
    split_idx = int(len(close) * TRAIN_RATIO)
    train = close.iloc[:split_idx]
    test = close.iloc[split_idx:]
    print(f"Train: {len(train)} días ({train.index[0].date()} — {train.index[-1].date()})")
    print(f"Test:  {len(test)} días ({test.index[0].date()} — {test.index[-1].date()})")
    return train, test


def fit_scaler(train: pd.Series) -> MinMaxScaler:
    """
    Ajusta el scaler ÚNICAMENTE sobre los datos de entrenamiento.
    CRÍTICO: nunca hacer fit sobre todo el dataset (data leakage).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values.reshape(-1, 1))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Scaler guardado en models/scaler.pkl")
    return scaler


def create_sequences(series: np.ndarray, seq_length: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma una serie 1D en pares (X, y) para LSTM.
    X[i] = ventana de seq_length días
    y[i] = precio del día siguiente
    """
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i - seq_length:i, 0])
        y.append(series[i, 0])
    return np.array(X).reshape(-1, seq_length, 1), np.array(y)


def decompose_series(data: pd.DataFrame) -> None:
    """Descomposición STL: tendencia, estacionalidad y residuo."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    close = data["Close"].asfreq("B").ffill()  # frecuencia días hábiles

    stl = STL(close, period=252, robust=True)  # 252 días hábiles = 1 año
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(close.index, result.observed, color="#1f77b4", lw=0.8)
    axes[0].set_title("Serie original")
    axes[1].plot(close.index, result.trend, color="#ff7f0e", lw=0.8)
    axes[1].set_title("Tendencia")
    axes[2].plot(close.index, result.seasonal, color="#2ca02c", lw=0.8)
    axes[2].set_title("Estacionalidad (anual)")
    axes[3].plot(close.index, result.resid, color="#d62728", lw=0.5, alpha=0.7)
    axes[3].set_title("Residuo")

    for ax in axes:
        ax.grid(alpha=0.3)
    plt.suptitle("Descomposición STL — IBEX 35", fontsize=13, y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / "stl_decomposition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Descomposición STL guardada: {path}")


def preprocess(data: pd.DataFrame) -> dict:
    """
    Pipeline completo de preprocesamiento.
    Retorna dict con X_train, y_train, X_test, y_test y scaler.
    """
    train, test = split_data(data)
    scaler = fit_scaler(train)

    train_scaled = scaler.transform(train.values.reshape(-1, 1))
    # Para test usamos los últimos seq_length días de train como contexto inicial
    full_scaled = scaler.transform(
        pd.concat([train, test]).values.reshape(-1, 1)
    )

    X_train, y_train = create_sequences(train_scaled)
    # El test necesita los últimos SEQUENCE_LENGTH días de train como contexto
    test_input = full_scaled[len(train_scaled) - SEQUENCE_LENGTH:]
    X_test, y_test = create_sequences(test_input)

    output = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_size": len(train),
        "test_size": len(test),
        "scaler": scaler,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "preprocessed_data.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    print(f"\nDatos preprocesados guardados en data/preprocessed_data.npz")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    return output
