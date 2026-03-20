import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"


def compare_models(arima_results: dict, lstm_results: dict,
                   test_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Genera tabla comparativa de métricas ARIMA vs LSTM."""
    comparison = pd.DataFrame({
        "Modelo": ["ARIMA(5,1,0)", "LSTM"],
        "MSE":  [round(arima_results["mse"], 2),  round(lstm_results["mse"], 2)],
        "MAE":  [round(arima_results["mae"], 2),  round(lstm_results["mae"], 2)],
        "RMSE": [round(arima_results["rmse"], 2), round(lstm_results["rmse"], 2)],
    })
    print("\n===== COMPARATIVA DE MODELOS =====")
    print(comparison.to_string(index=False))
    return comparison


def plot_predictions(actuals: np.ndarray, arima_preds: np.ndarray,
                     lstm_preds: np.ndarray, test_index: pd.DatetimeIndex) -> None:
    """Grafica predicciones de ambos modelos vs valores reales."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mostrar solo últimos 250 días para mejor legibilidad
    n = min(250, len(actuals))
    idx = test_index[-n:]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(idx, actuals[-n:],    color="#1f77b4", lw=1.2, label="Real")
    ax.plot(idx, arima_preds[-n:], color="#ff7f0e", lw=1, ls="--", label="ARIMA")
    ax.plot(idx, lstm_preds[-n:],  color="#2ca02c", lw=1, label="LSTM")
    ax.set_title("IBEX 35 — Predicciones vs Valores Reales (últimos 250 días de test)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Puntos")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "predictions_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Grafico comparativo guardado: {path}")


def plot_metrics_bar(comparison: pd.DataFrame) -> None:
    """Gráfico de barras comparando RMSE y MAE."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#ff7f0e", "#2ca02c"]

    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        bars = ax.bar(comparison["Modelo"], comparison[metric], color=colors, alpha=0.85)
        ax.set_title(f"Comparativa {metric}")
        ax.set_ylabel("Puntos del índice")
        ax.grid(alpha=0.3, axis="y")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{bar.get_height():.0f}",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = PLOTS_DIR / "metrics_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Grafico de metricas guardado: {path}")
