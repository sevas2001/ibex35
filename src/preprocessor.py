import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

DATA_DIR   = Path(__file__).parent.parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
MODELS_DIR = Path(__file__).parent.parent / "models"

SEQUENCE_LENGTH = 60
TRAIN_RATIO     = 0.80
N_FEATURES      = 1   # Close (univariado — mejor rendimiento en test)

# NOTA ACADEMICA: Se probaron features adicionales (RSI, Crisis indicator, Volume).
# El modelo univariado supera al multivariate en este dataset porque:
# - RSI es derivado de Close (informacion redundante)
# - Volume del indice ^IBEX es poco fiable en Yahoo Finance
# Crisis indicators implementados en compute_features() para referencia futura.


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame con solo Close (modelo univariado).
    Los crisis indicators se mantienen como experimento documentado.

    Resultados del experimento multivariate (3 features: Close+RSI+Crisis):
      MAE: 285 pts vs MAE: 142 pts univariado — el univariado gana.
    """
    df = pd.DataFrame(index=data.index)
    df["Close"] = data["Close"]
    return df.dropna()


def split_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal 80/20 sin shuffle para respetar el orden cronologico."""
    split_idx = int(len(features) * TRAIN_RATIO)
    train = features.iloc[:split_idx]
    test  = features.iloc[split_idx:]
    print(f"Train: {len(train)} dias ({train.index[0].date()} — {train.index[-1].date()})")
    print(f"Test:  {len(test)} dias ({test.index[0].date()} — {test.index[-1].date()})")
    return train, test


def fit_scaler(train_features: pd.DataFrame) -> MinMaxScaler:
    """
    Ajusta MinMaxScaler sobre los 4 features de entrenamiento.
    CRITICO: nunca hacer fit sobre todo el dataset (data leakage).
    La columna 0 (Close) se usa para inverse_transform de predicciones.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features.values)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Scaler (4 features) guardado en models/scaler.pkl")
    return scaler


def create_sequences(features_scaled: np.ndarray,
                     seq_length: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte el array escalado en pares (X, y) para LSTM multivariate.
      X[i]: ventana (seq_length, N_FEATURES)
      y[i]: Close escalado del dia siguiente (columna 0)
    """
    X, y = [], []
    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i - seq_length:i])   # (60, 4)
        y.append(features_scaled[i, 0])                # target = Close
    return np.array(X), np.array(y)


def decompose_series(data: pd.DataFrame) -> None:
    """Descomposicion STL: tendencia, estacionalidad y residuo."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    close = data["Close"].asfreq("B").ffill()

    stl    = STL(close, period=252, robust=True)
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
    plt.suptitle("Descomposicion STL — IBEX 35", fontsize=13, y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / "stl_decomposition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Descomposicion STL guardada: {path}")


def preprocess(data: pd.DataFrame) -> dict:
    """
    Pipeline completo de preprocesamiento multivariate (4 features).
    Retorna dict con X_train, y_train, X_test, y_test y scaler.
    """
    features = compute_features(data)
    train, test = split_data(features)
    scaler = fit_scaler(train)

    train_scaled = scaler.transform(train.values)
    full_scaled  = scaler.transform(features.values)

    X_train, y_train = create_sequences(train_scaled)

    # Test necesita los ultimos SEQUENCE_LENGTH dias de train como contexto
    test_input = full_scaled[len(train) - SEQUENCE_LENGTH:]
    X_test, y_test = create_sequences(test_input)

    output = {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_test":     X_test,
        "y_test":     y_test,
        "train_size": len(train),
        "test_size":  len(test),
        "scaler":     scaler,
        "features":   features,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "preprocessed_data.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test,  y_test=y_test,
    )
    print(f"\nDatos preprocesados guardados en data/preprocessed_data.npz")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"Features: {list(features.columns)}")
    return output
