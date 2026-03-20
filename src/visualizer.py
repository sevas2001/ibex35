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
