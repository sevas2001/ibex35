"""
backtester.py — Backtesting dia a dia sobre el conjunto de test.

Para cada dia t del test set:
  - Toma los 60 dias previos (ventana deslizante) como input
  - Predice el precio de cierre de t+1 con LSTM, GRU y ARIMA
  - Compara con el precio real de t+1

Sin data leakage: los modelos NO se reentrenan durante el backtest.
El scaler ya fue ajustado solo sobre el train set.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"
DATA_DIR  = Path(__file__).parent.parent / "data"
SEQ_LEN   = 60


def _inv(scaler, scaled_val: float) -> float:
    """Inverse-transform un valor escalar de Close (N_FEATURES=1)."""
    dummy = np.array([[scaled_val]])
    return float(scaler.inverse_transform(dummy)[0, 0])


def run_backtest(
    full_close: np.ndarray,
    full_index: pd.DatetimeIndex,
    train_size: int,
    scaler,
    lstm_model,
    gru_model,
    arima_order: tuple = (5, 1, 0),
    full_history: bool = False,
) -> pd.DataFrame:
    """
    Ejecuta backtesting dia a dia.

    full_history=False  → solo test set (out-of-sample, sin leakage)
    full_history=True   → toda la serie desde el dia 60
                          (train=in-sample, test=out-of-sample, marcados en el grafico)

    Parametros
    ----------
    full_close   : serie completa de Close (sin escalar), shape (N,)
    full_index   : indice de fechas de TODA la serie
    train_size   : numero de dias de entrenamiento
    scaler       : MinMaxScaler ajustado sobre train
    lstm_model   : modelo LSTM de Keras
    gru_model    : modelo GRU de Keras (puede ser None)
    arima_order  : orden ARIMA, por defecto (5,1,0)
    full_history : True = corre desde dia 60, False = solo test set
    """
    scaled_full = scaler.transform(full_close.reshape(-1, 1)).flatten()

    # Rango de indices a evaluar
    if full_history:
        start_idx = SEQ_LEN          # primer dia con ventana completa
        eval_index = full_index[start_idx:]
        print(f"\nBacktesting COMPLETO: {len(eval_index)} dias ({eval_index[0].date()} -> {eval_index[-1].date()})...")
    else:
        start_idx = train_size
        eval_index = full_index[train_size:]
        print(f"\nBacktesting test set: {len(eval_index)} dias...")

    results = []
    n = len(eval_index)

    for i, fecha in enumerate(eval_index):
        abs_idx = start_idx + i

        real = full_close[abs_idx]
        prev = full_close[abs_idx - 1]

        # ── LSTM ─────────────────────────────────────────────
        lstm_pred = None
        if lstm_model is not None:
            window = scaled_full[abs_idx - SEQ_LEN : abs_idx]
            X = window.reshape(1, SEQ_LEN, 1)
            p_scaled = float(lstm_model.predict(X, verbose=0)[0, 0])
            lstm_pred = _inv(scaler, p_scaled)

        # ── GRU ──────────────────────────────────────────────
        gru_pred = None
        if gru_model is not None:
            window = scaled_full[abs_idx - SEQ_LEN : abs_idx]
            X = window.reshape(1, SEQ_LEN, 1)
            p_scaled = float(gru_model.predict(X, verbose=0)[0, 0])
            gru_pred = _inv(scaler, p_scaled)

        # ── ARIMA rolling (ventana deslizante 200 dias) ──────
        arima_pred = None
        try:
            arima_window = full_close[max(0, abs_idx - 200) : abs_idx]
            fit = ARIMA(arima_window, order=arima_order).fit()
            arima_pred = float(fit.forecast(steps=1)[0])
        except Exception:
            arima_pred = float(prev)

        # in_sample: True si estamos en el periodo de entrenamiento
        in_sample = (abs_idx < train_size)

        row = {
            "fecha":       fecha,
            "real":        round(real, 2),
            "lstm":        round(lstm_pred, 2) if lstm_pred else None,
            "gru":         round(gru_pred,  2) if gru_pred  else None,
            "arima":       round(arima_pred, 2),
            "error_lstm":  round(abs(real - lstm_pred), 2) if lstm_pred else None,
            "error_gru":   round(abs(real - gru_pred),  2) if gru_pred  else None,
            "error_arima": round(abs(real - arima_pred), 2),
            "dir_lstm":    int((lstm_pred > prev) == (real > prev)) if lstm_pred else None,
            "dir_gru":     int((gru_pred  > prev) == (real > prev)) if gru_pred  else None,
            "dir_arima":   int((arima_pred > prev) == (real > prev)),
            "in_sample":   in_sample,
        }
        results.append(row)

        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"  {i+1}/{n} dias procesados...")

    df = pd.DataFrame(results).set_index("fecha")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / ("backtest_full.csv" if full_history else "backtest_results.csv")
    df.to_csv(out_path)
    print(f"\nBacktest guardado: {out_path}")

    _print_metrics(df)
    return df


def _print_metrics(df: pd.DataFrame) -> None:
    """Imprime tabla de metricas por modelo."""
    print("\n========== BACKTEST — METRICAS FINALES ==========")
    header = f"{'Modelo':<14} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'Dir Acc':>9}"
    print(header)
    print("-" * len(header))

    for modelo, ecol, dcol in [
        ("ARIMA(5,1,0)", "error_arima", "dir_arima"),
        ("LSTM",         "error_lstm",  "dir_lstm"),
        ("GRU",          "error_gru",   "dir_gru"),
    ]:
        sub = df[[ecol, dcol, "real"]].dropna()
        if sub.empty:
            continue
        mae  = round(sub[ecol].mean(), 2)
        rmse = round(float(np.sqrt((sub[ecol] ** 2).mean())), 2)
        mape = round(float((sub[ecol] / sub["real"]).mean() * 100), 2)
        dacc = round(sub[dcol].mean() * 100, 1)
        print(f"{modelo:<14} {mae:>8.2f} {rmse:>8.2f} {mape:>7.2f}% {dacc:>8.1f}%")

    print("=================================================")


def plot_backtest(df: pd.DataFrame, train_cutoff_date=None) -> None:
    """Genera grafico completo del backtest: precios reales vs modelos.

    train_cutoff_date: si se pasa, dibuja linea vertical separando in-sample / out-of-sample.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    full = train_cutoff_date is not None
    fname = "backtest_full.png" if full else "backtest_comparison.png"
    title_suffix = "1993–2026 (in-sample + out-of-sample)" if full else "test set completo"

    fig, axes = plt.subplots(2, 1, figsize=(20, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0a0a0f")
    for ax in axes:
        ax.set_facecolor("#111118")
        for spine in ax.spines.values():
            spine.set_color("#222230")

    # ── Panel superior: precios ───────────────────────────────
    ax = axes[0]
    ax.plot(df.index, df["real"],  color="#00d4aa", lw=1.2, label="Real",          zorder=4)
    ax.plot(df.index, df["arima"], color="#fb923c", lw=0.8, ls="--", label="ARIMA", alpha=0.8, zorder=3)
    if "lstm" in df and df["lstm"].notna().any():
        ax.plot(df.index, df["lstm"],  color="#3b82f6", lw=0.8, label="LSTM",  alpha=0.8, zorder=3)
    if "gru" in df and df["gru"].notna().any():
        ax.plot(df.index, df["gru"],   color="#8b5cf6", lw=0.8, ls=":", label="GRU", alpha=0.8, zorder=3)

    # Linea de corte train/test
    if train_cutoff_date is not None:
        for a in axes:
            a.axvline(x=train_cutoff_date, color="#facc15", lw=1.0, ls="--", alpha=0.7, zorder=5)
        ax.text(train_cutoff_date, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 15000,
                "  Train / Test", color="#facc15", fontsize=8, va="top", alpha=0.8)

    ax.set_title(f"IBEX 35 — Backtest: Predicciones vs Valores Reales ({title_suffix})",
                 color="#f1f5f9", fontsize=13, pad=10)
    ax.set_ylabel("Puntos", color="#94a3b8")
    ax.tick_params(colors="#475569")
    ax.grid(color="#1e1e2e", linewidth=0.5)
    ax.legend(facecolor="#16161f", edgecolor="#333", labelcolor="#f1f5f9", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # ── Panel inferior: errores absolutos ────────────────────
    ax2 = axes[1]
    ax2.fill_between(df.index, df["error_arima"], alpha=0.4, color="#fb923c", label="Error ARIMA")
    if "error_lstm" in df and df["error_lstm"].notna().any():
        ax2.fill_between(df.index, df["error_lstm"], alpha=0.4, color="#3b82f6", label="Error LSTM")
    ax2.set_ylabel("Error abs.", color="#94a3b8")
    ax2.set_xlabel("Fecha", color="#94a3b8")
    ax2.tick_params(colors="#475569")
    ax2.grid(color="#1e1e2e", linewidth=0.5)
    ax2.legend(facecolor="#16161f", edgecolor="#333", labelcolor="#f1f5f9", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout(rect=[0, 0, 1, 1])
    path = PLOTS_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    print(f"Grafico backtest guardado: {path}")
