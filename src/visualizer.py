import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"


def run_eda(data: pd.DataFrame) -> None:
    """Imprime un resumen exploratorio del dataset."""
    print("\n====== EDA: IBEX 35 ======")
    print(f"Shape:      {data.shape}")
    print(f"Desde:      {data.index[0].date()}")
    print(f"Hasta:      {data.index[-1].date()}")
    print(f"\nNulos:\n{data.isnull().sum()}")
    print(f"\nEstadísticas descriptivas:\n{data['Close'].describe().round(2)}")


def plot_close_price(data: pd.DataFrame) -> None:
    """Gráfico del precio de cierre histórico."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data.index, data["Close"], color="#1f77b4", linewidth=0.8)
    ax.set_title("IBEX 35 — Precio de Cierre Histórico (2012–hoy)", fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Puntos")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "close_price_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {path}")


def plot_volume(data: pd.DataFrame) -> None:
    """Gráfico del volumen de negociación."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.bar(data.index, data["Volume"], color="#aec7e8", alpha=0.7, width=1)
    ax.set_title("IBEX 35 — Volumen de Negociación", fontsize=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Volumen")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    path = PLOTS_DIR / "volume_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {path}")


def plot_train_fit(model, X_train: np.ndarray, y_train: np.ndarray,
                   scaler, data_index: pd.DatetimeIndex) -> None:
    """
    Grafica predicciones del modelo sobre el TRAIN SET vs valores reales.
    Si el modelo aprendió bien, las curvas deben solaparse.
    Guarda en data/plots/train_fit.png
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_scaled = model.predict(X_train, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actuals = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    seq_len = X_train.shape[1]
    idx = data_index[seq_len: seq_len + len(actuals)]
    n = min(500, len(idx))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(idx[-n:], actuals[-n:], color="#1f77b4", lw=1.2, label="Real (train)")
    axes[0].plot(idx[-n:], preds[-n:], color="#2ca02c", lw=1, ls="--",
                 label="Prediccion LSTM (train)")
    axes[0].set_title("Ajuste del modelo LSTM sobre datos de entrenamiento")
    axes[0].set_ylabel("Puntos IBEX 35")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    residuals = actuals[-n:] - preds[-n:]
    axes[1].fill_between(idx[-n:], residuals, alpha=0.5, color="#9467bd")
    axes[1].axhline(0, color="white", lw=0.8, ls="--")
    axes[1].set_title("Residuos (Real - Prediccion)")
    axes[1].set_ylabel("Error (pts)")
    axes[1].grid(alpha=0.3)

    mae = round(np.mean(np.abs(actuals - preds)), 2)
    rmse = round(np.sqrt(np.mean((actuals - preds) ** 2)), 2)
    fig.suptitle(f"Train Fit — MAE: {mae} pts | RMSE: {rmse} pts", fontsize=12)

    plt.tight_layout()
    path = PLOTS_DIR / "train_fit.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Train fit guardado: {path} | MAE train: {mae} | RMSE train: {rmse}")
