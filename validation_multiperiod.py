"""
validation_multiperiod.py — Validacion con multiples periodos de corte

Entrena ARIMA sobre datos hasta cada corte y evalua los 6 meses siguientes.
Cortes: 2015-01-01, 2017-01-01, 2019-01-01, 2021-01-01, 2023-01-01
Sin re-entrenar redes neuronales — solo ARIMA rolling (rapido).
Demuestra consistencia del modelo en distintos regimenes de mercado.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from scipy import stats

DATA_PATH = Path("data/raw/ibex35_full.csv")
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(exist_ok=True)

DARK_BG = "#0a0a0f"; SURFACE  = "#111118"; SURFACE2 = "#16161f"
ORANGE  = "#fb923c"; GREEN    = "#00d4aa"; BLUE     = "#3b82f6"
YELLOW  = "#facc15"; TEXT     = "#f1f5f9"; MUTED    = "#94a3b8"
GRID    = "#1e1e2e"; RED      = "#f87171"; PURPLE   = "#8b5cf6"

CORTES = [
    ("2015-01-01", "Mercado lateral post-crisis"),
    ("2017-01-01", "Expansion pre-Brexit"),
    ("2019-01-01", "Pre-COVID"),
    ("2021-01-01", "Recuperacion post-COVID"),
    ("2023-01-01", "Mercado actual"),
]
COLORES = [ORANGE, GREEN, BLUE, PURPLE, YELLOW]
VENTANA_TEST = 126   # ~6 meses de dias habiles

print("=" * 65)
print("  VALIDACION MULTI-PERIODO — ARIMA(5,1,0)")
print("  Cortes: 2015, 2017, 2019, 2021, 2023")
print("=" * 65)

# ── Cargar datos ──────────────────────────────────────────────────────────────
full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
full = full[["Close"]].dropna()

# Ampliar con yfinance si posible
try:
    import yfinance as yf
    from datetime import datetime, timedelta
    end = datetime.now() + timedelta(days=1)
    rec = yf.download("^IBEX", start="2026-01-01",
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    if isinstance(rec.columns, pd.MultiIndex):
        rec.columns = rec.columns.get_level_values(0)
    if not rec.empty:
        rec = rec[["Close"]].dropna()
        full = pd.concat([full, rec])
        full = full[~full.index.duplicated(keep="last")].sort_index()
except Exception:
    pass

print(f"Datos: {len(full)} dias | {full.index[0].date()} -> {full.index[-1].date()}\n")


# ── Funcion: evalua ARIMA rolling sobre un periodo de test ───────────────────
def evaluar_periodo(close_train: pd.Series,
                    close_test:  pd.Series,
                    label: str) -> dict:
    """
    Rolling one-step-ahead ARIMA(5,1,0).
    En cada dia de test: ajusta sobre los ultimos 200 dias disponibles
    (train + dias de test ya conocidos), predice el siguiente.
    """
    window = close_train.values.tolist()
    preds, reals = [], []

    for i, (idx, real) in enumerate(close_test.items()):
        # Predecir
        try:
            fit  = ARIMAModel(window[-200:], order=(5,1,0)).fit()
            pred = float(fit.forecast(steps=1).iloc[0])
        except Exception:
            pred = window[-1]

        preds.append(pred)
        reals.append(float(real))
        window.append(float(real))   # añadir precio real antes del siguiente dia

        if (i+1) % 30 == 0:
            print(f"  [{label}] {i+1}/{len(close_test)} dias...")

    preds = np.array(preds)
    reals = np.array(reals)
    errors = np.abs(preds - reals)
    dirs   = np.array([
        1 if (reals[i] > reals[i-1]) == (preds[i] > reals[i-1]) else 0
        for i in range(1, len(reals))
    ])

    # Naive: pred = precio anterior
    naive_err = np.abs(np.diff(reals))

    # Diebold-Mariano vs naive
    d = errors[1:]**2 - naive_err**2
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-10)
    dm_p    = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "label":    label,
        "n":        len(reals),
        "mae":      errors.mean(),
        "rmse":     np.sqrt((errors**2).mean()),
        "mape":     (errors / reals * 100).mean(),
        "dir_acc":  dirs.mean() * 100,
        "dm_stat":  dm_stat,
        "dm_p":     dm_p,
        "preds":    preds,
        "reals":    reals,
        "index":    close_test.index,
        "mae_ci_lo": errors.mean() - 1.96 * errors.std(ddof=1) / np.sqrt(len(errors)),
        "mae_ci_hi": errors.mean() + 1.96 * errors.std(ddof=1) / np.sqrt(len(errors)),
    }


# ── Evaluar todos los cortes ──────────────────────────────────────────────────
resultados = []

for (corte, descripcion), color in zip(CORTES, COLORES):
    corte_ts = pd.Timestamp(corte)
    train    = full[full.index <  corte_ts]["Close"]
    test_all = full[full.index >= corte_ts]["Close"]
    test     = test_all.iloc[:VENTANA_TEST]

    if len(test) < 10:
        print(f"  Saltando {corte}: pocos datos de test ({len(test)})")
        continue

    print(f"\n--- Corte: {corte} ({descripcion}) ---")
    print(f"  Train: {len(train)} dias | Test: {len(test)} dias "
          f"({test.index[0].date()} -> {test.index[-1].date()})")

    res = evaluar_periodo(train, test, corte)
    res["corte"]       = corte
    res["descripcion"] = descripcion
    res["color"]       = color
    resultados.append(res)


# ── Tabla de resultados ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  TABLA COMPARATIVA — ARIMA(5,1,0) por periodo")
print("=" * 80)
print(f"\n  {'Corte':<12} {'Descripcion':<30} {'n':>4}  "
      f"{'MAE':>7} {'IC95%':>16} {'RMSE':>7} {'MAPE%':>6} {'Dir%':>6} {'DM_p':>8}")
print("  " + "-"*78)

for r in resultados:
    dm_sig = "***" if r["dm_p"] < 0.001 else ("**" if r["dm_p"] < 0.01 else
             ("*" if r["dm_p"] < 0.05 else "ns"))
    print(f"  {r['corte']:<12} {r['descripcion']:<30} {r['n']:>4}  "
          f"{r['mae']:>7.1f} [{r['mae_ci_lo']:>6.1f},{r['mae_ci_hi']:>6.1f}]  "
          f"{r['rmse']:>7.1f} {r['mape']:>6.2f}% {r['dir_acc']:>5.1f}%  "
          f"p={r['dm_p']:.4f}{dm_sig}")

print("\n  *** p<0.001  ** p<0.01  * p<0.05  ns=no significativo")

# MAE medio global
mae_global = np.mean([r["mae"] for r in resultados])
print(f"\n  MAE medio global (5 periodos): {mae_global:.1f} pts")
print(f"  MAE backtest OOS original:     85.0 pts")

# Guardar CSV
rows = [{
    "corte": r["corte"], "descripcion": r["descripcion"],
    "n": r["n"], "mae": round(r["mae"],1), "rmse": round(r["rmse"],1),
    "mape": round(r["mape"],2), "dir_acc": round(r["dir_acc"],1),
    "dm_p": round(r["dm_p"],4),
    "mae_ci_lo": round(r["mae_ci_lo"],1), "mae_ci_hi": round(r["mae_ci_hi"],1),
} for r in resultados]
pd.DataFrame(rows).to_csv("data/validation_multiperiod.csv", index=False)
print("\nCSV guardado: data/validation_multiperiod.csv")


# ── GRAFICO 1: Panel de predicciones por periodo ──────────────────────────────
print("\nGenerando graficos...")

fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
fig.suptitle("ARIMA(5,1,0) — Validacion en 5 periodos de mercado distintos\n"
             "Prediccion rolling one-step-ahead | 6 meses por periodo",
             color=TEXT, fontsize=13, y=0.98)

n_cols = 3
n_rows = 2
gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.3)

for idx, r in enumerate(resultados):
    row, col = divmod(idx, n_cols)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, lw=0.4)
    for sp in ax.spines.values(): sp.set_color("#222230")

    ax.plot(r["index"], r["reals"], color=TEXT,      lw=1.2, label="Real")
    ax.plot(r["index"], r["preds"], color=r["color"], lw=1,
            alpha=0.85, label=f"ARIMA pred")

    dm_sig = "***" if r["dm_p"] < 0.001 else ("**" if r["dm_p"] < 0.01 else
             ("*" if r["dm_p"] < 0.05 else "ns"))
    ax.set_title(
        f"{r['corte']} — {r['descripcion']}\n"
        f"MAE={r['mae']:.1f}  MAPE={r['mape']:.2f}%  Dir={r['dir_acc']:.0f}%  DM:{dm_sig}",
        color=TEXT, fontsize=8.5, pad=6)
    ax.set_ylabel("Puntos IBEX", color=MUTED, fontsize=7)
    ax.legend(facecolor="#16161f", edgecolor="#333", labelcolor=TEXT,
              fontsize=7, loc="upper left")

# Panel resumen (posicion 6)
ax_sum = fig.add_subplot(gs[1, 2])
ax_sum.set_facecolor(SURFACE)
ax_sum.tick_params(colors=MUTED, labelsize=8)
ax_sum.grid(color=GRID, lw=0.4, axis="y")
for sp in ax_sum.spines.values(): sp.set_color("#222230")

maes   = [r["mae"]  for r in resultados]
cis_lo = [r["mae_ci_lo"] for r in resultados]
cis_hi = [r["mae_ci_hi"] for r in resultados]
colors = [r["color"] for r in resultados]
labels = [r["corte"][:4] for r in resultados]
x = np.arange(len(resultados))

bars = ax_sum.bar(x, maes, color=colors, alpha=0.85, width=0.6)
ax_sum.errorbar(x, maes,
                yerr=[np.array(maes)-np.array(cis_lo),
                      np.array(cis_hi)-np.array(maes)],
                fmt="none", color=TEXT, capsize=5, lw=1.5)
ax_sum.axhline(85.0, color=YELLOW, lw=1.2, ls="--",
               label="MAE OOS original (85 pts)")
ax_sum.axhline(mae_global, color=GREEN, lw=1, ls=":",
               label=f"MAE medio ({mae_global:.1f} pts)")

for bar, mae in zip(bars, maes):
    ax_sum.text(bar.get_x() + bar.get_width()/2, mae + 3,
                f"{mae:.0f}", ha="center", color=TEXT, fontsize=8)

ax_sum.set_xticks(x)
ax_sum.set_xticklabels(labels, color=MUTED, fontsize=8)
ax_sum.set_title("MAE por periodo\n(barras de error = IC95%)",
                 color=TEXT, fontsize=9)
ax_sum.set_ylabel("MAE (pts)", color=MUTED, fontsize=8)
ax_sum.legend(facecolor="#16161f", edgecolor="#333",
              labelcolor=TEXT, fontsize=7)

plt.savefig(PLOTS_DIR / "validation_multiperiod.png",
            dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Guardado: validation_multiperiod.png")


# ── GRAFICO 2: MAE rolling 30 dias sobre backtest full ───────────────────────
bt = pd.read_csv("data/backtest_full.csv", index_col="fecha", parse_dates=True)
bt["in_sample"] = bt["in_sample"].astype(str).str.strip().str.lower() == "true"

fig2, ax = plt.subplots(figsize=(18, 5), facecolor=DARK_BG)
ax.set_facecolor(SURFACE)
ax.tick_params(colors=MUTED, labelsize=8)
ax.grid(color=GRID, lw=0.4)
for sp in ax.spines.values(): sp.set_color("#222230")

mae_roll = bt["error_arima"].abs().rolling(60, min_periods=20).mean()
ax.plot(bt.index, mae_roll, color=ORANGE, lw=1, alpha=0.9,
        label="MAE rolling 60 dias (ARIMA)")
ax.axhline(85.0, color=YELLOW, lw=1, ls="--", alpha=0.8,
           label="MAE OOS global=85 pts")
ax.axvline(bt[~bt["in_sample"]].index[0], color=GREEN, lw=1.2, ls="--",
           label="Inicio OOS (~2019)")

# Marcar los cortes del experimento
for (corte, desc), color in zip(CORTES, COLORES):
    ts = pd.Timestamp(corte)
    if ts in bt.index or (ts > bt.index[0] and ts < bt.index[-1]):
        ax.axvline(ts, color=color, lw=0.8, ls=":", alpha=0.7)
        ax.text(ts, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 300,
                f" {corte[:4]}", color=color, fontsize=7, va="top")

# Marcar crisis
for label, fecha, c in [("Crisis 2008","2007-01-01",RED),
                         ("COVID","2020-03-01","#a78bfa")]:
    ax.axvspan(pd.Timestamp(fecha),
               pd.Timestamp("2010-01-01") if "Crisis" in label else pd.Timestamp("2021-06-01"),
               alpha=0.06, color=c)
    ax.text(pd.Timestamp(fecha), 50, f" {label}", color=c, fontsize=7)

ax.set_title("MAE rolling 60 dias — ARIMA(5,1,0) sobre historia completa 1993-2026\n"
             "Lineas punteadas = cortes del experimento multi-periodo",
             color=TEXT, fontsize=11)
ax.set_ylabel("MAE (pts)", color=MUTED)
ax.set_ylim(bottom=0)
ax.legend(facecolor="#16161f", edgecolor="#333", labelcolor=TEXT, fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "validation_mae_rolling_full.png",
            dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Guardado: validation_mae_rolling_full.png")

print("\nValidacion multi-periodo completada.")
