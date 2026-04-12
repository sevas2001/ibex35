"""
regression_analysis.py — Análisis de regresión completo sobre backtest_full.csv
Modelos: ARIMA, LSTM, GRU vs precio real del IBEX 35
Separado por periodo: in-sample (1993-2019) y out-of-sample (2019-2026)
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

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# ── Configuración ────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG   = "#0a0a0f"
SURFACE   = "#111118"
GREEN     = "#00d4aa"
BLUE      = "#3b82f6"
PURPLE    = "#8b5cf6"
ORANGE    = "#fb923c"
TEXT      = "#f1f5f9"
MUTED     = "#94a3b8"
GRID      = "#1e1e2e"

MODELOS = [
    ("ARIMA",  "arima",  "error_arima", ORANGE),
    ("LSTM",   "lstm",   "error_lstm",  BLUE),
    ("GRU",    "gru",    "error_gru",   PURPLE),
]

PERIODOS = [
    ("Completo",       None),
    ("In-sample",      True),
    ("Out-of-sample",  False),
]


# ── Carga de datos ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "backtest_full.csv", index_col="fecha", parse_dates=True)
df["in_sample"] = df["in_sample"].astype(str).str.strip().str.lower() == "true"
print(f"Datos cargados: {len(df)} filas | {df.index[0].date()} → {df.index[-1].date()}")
print(f"  In-sample:      {df['in_sample'].sum()} días")
print(f"  Out-of-sample:  {(~df['in_sample']).sum()} días\n")


# ── Función principal de regresión ───────────────────────────────────────────
def regression_stats(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """OLS: y_real ~ β0 + β1·y_pred + ε"""
    mask = ~(np.isnan(y_real) | np.isnan(y_pred))
    y, yp = y_real[mask], y_pred[mask]
    n = len(y)

    X = sm.add_constant(yp)
    res = sm.OLS(y, X).fit()

    beta0, beta1 = res.params
    ci = res.conf_int(alpha=0.05)
    pvals = res.pvalues
    resid = res.resid
    sigma = np.sqrt(res.mse_resid)

    # R² ajustado
    r2     = res.rsquared
    r2_adj = res.rsquared_adj

    # Durbin-Watson
    dw = durbin_watson(resid)

    # Breusch-Pagan
    bp_lm, bp_pval, _, _ = het_breuschpagan(resid, X)

    # Normalidad (KS si n>5000, Shapiro si n<=5000)
    if n <= 5000:
        norm_stat, norm_pval = stats.shapiro(resid)
        norm_test = "Shapiro-Wilk"
    else:
        norm_stat, norm_pval = stats.kstest(resid, "norm",
                                             args=(resid.mean(), resid.std()))
        norm_test = "Kolmogorov-Smirnov"

    # ci puede ser ndarray o DataFrame segun version de statsmodels
    ci_arr = np.array(ci)
    pvals_arr = np.array(pvals)

    return {
        "n":          n,
        "beta0":      beta0,
        "beta0_ci":   (ci_arr[0, 0], ci_arr[0, 1]),
        "beta0_pval": pvals_arr[0],
        "beta1":      beta1,
        "beta1_ci":   (ci_arr[1, 0], ci_arr[1, 1]),
        "beta1_pval": pvals_arr[1],
        "r2":         r2,
        "r2_adj":     r2_adj,
        "sigma":      sigma,
        "dw":         dw,
        "bp_stat":    bp_lm,
        "bp_pval":    bp_pval,
        "norm_test":  norm_test,
        "norm_stat":  norm_stat,
        "norm_pval":  norm_pval,
        "resid":      resid,
        "y":          y,
        "yp":         yp,
    }


# ── Calcular y mostrar ────────────────────────────────────────────────────────
resultados = {}  # (modelo, periodo) → dict

print("=" * 80)
print("  ANÁLISIS DE REGRESIÓN — IBEX 35 BACKTEST")
print("=" * 80)

header = (f"\n{'Modelo':<8} {'Periodo':<18} {'n':>5}  "
          f"{'β₀':>9} {'β₁':>8} {'R²':>7} {'R²adj':>7} "
          f"{'σ':>8} {'DW':>6} {'BP_p':>7} {'Norm_p':>8}")
print(header)
print("-" * len(header))

for nombre, pred_col, _, color in MODELOS:
    for periodo_nombre, in_samp in PERIODOS:
        if in_samp is None:
            sub = df
        else:
            sub = df[df["in_sample"] == in_samp]

        y_real = sub["real"].values.astype(float)
        y_pred = sub[pred_col].values.astype(float)

        stats_dict = regression_stats(y_real, y_pred)
        resultados[(nombre, periodo_nombre)] = stats_dict

        # Indicadores de significancia
        sig_b0 = "***" if stats_dict["beta0_pval"] < 0.001 else ("**" if stats_dict["beta0_pval"] < 0.01 else ("*" if stats_dict["beta0_pval"] < 0.05 else "  "))
        sig_b1 = "***" if stats_dict["beta1_pval"] < 0.001 else ("**" if stats_dict["beta1_pval"] < 0.01 else ("*" if stats_dict["beta1_pval"] < 0.05 else "  "))

        print(f"{nombre:<8} {periodo_nombre:<18} {stats_dict['n']:>5}  "
              f"{stats_dict['beta0']:>9.1f} "
              f"{stats_dict['beta1']:>7.4f}{sig_b1}  "
              f"{stats_dict['r2']:>6.4f} {stats_dict['r2_adj']:>7.4f}  "
              f"{stats_dict['sigma']:>7.1f}  "
              f"{stats_dict['dw']:>5.3f}  "
              f"{stats_dict['bp_pval']:>7.4f}  "
              f"{stats_dict['norm_pval']:>8.4f}")

print("\n  *** p<0.001  ** p<0.01  * p<0.05\n")

# Detalle por modelo
for nombre, pred_col, _, _ in MODELOS:
    print(f"\n{'─'*60}")
    print(f"  {nombre} — Detalle completo")
    print(f"{'─'*60}")
    for periodo_nombre, _ in PERIODOS:
        s = resultados[(nombre, periodo_nombre)]
        print(f"\n  [{periodo_nombre}]  n={s['n']}")
        print(f"    β₀ = {s['beta0']:>10.3f}  IC95%: [{s['beta0_ci'][0]:.3f}, {s['beta0_ci'][1]:.3f}]  p={s['beta0_pval']:.4f}")
        print(f"    β₁ = {s['beta1']:>10.6f}  IC95%: [{s['beta1_ci'][0]:.6f}, {s['beta1_ci'][1]:.6f}]  p={s['beta1_pval']:.4f}")
        print(f"    R²={s['r2']:.5f}  R²adj={s['r2_adj']:.5f}  σ={s['sigma']:.2f}")
        print(f"    Durbin-Watson={s['dw']:.4f}  (ref: ~2 = sin autocorr)")
        print(f"    Breusch-Pagan: stat={s['bp_stat']:.3f}  p={s['bp_pval']:.4f}  "
              f"({'heteroced.' if s['bp_pval']<0.05 else 'homoced.'})")
        print(f"    {s['norm_test']}: stat={s['norm_stat']:.4f}  p={s['norm_pval']:.4f}  "
              f"({'NO normal' if s['norm_pval']<0.05 else 'normal'})")


# ── Tabla resumen CSV ─────────────────────────────────────────────────────────
rows = []
for (nombre, periodo), s in resultados.items():
    rows.append({
        "modelo":      nombre,
        "periodo":     periodo,
        "n":           s["n"],
        "beta0":       round(s["beta0"], 4),
        "beta0_ci_lo": round(s["beta0_ci"][0], 4),
        "beta0_ci_hi": round(s["beta0_ci"][1], 4),
        "beta0_pval":  round(s["beta0_pval"], 6),
        "beta1":       round(s["beta1"], 6),
        "beta1_ci_lo": round(s["beta1_ci"][0], 6),
        "beta1_ci_hi": round(s["beta1_ci"][1], 6),
        "beta1_pval":  round(s["beta1_pval"], 6),
        "r2":          round(s["r2"], 5),
        "r2_adj":      round(s["r2_adj"], 5),
        "sigma":       round(s["sigma"], 3),
        "durbin_watson": round(s["dw"], 4),
        "bp_pval":     round(s["bp_pval"], 6),
        "norm_pval":   round(s["norm_pval"], 6),
        "norm_test":   s["norm_test"],
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(DATA_DIR / "regression_summary.csv", index=False)
print(f"\nTabla resumen guardada: data/regression_summary.csv")


# ── Gráficos por modelo (3×3 grid) ────────────────────────────────────────────
print("\nGenerando gráficos de diagnóstico...")

for nombre, pred_col, _, color in MODELOS:
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle(f"IBEX 35 — Diagnóstico de Regresión: {nombre}",
                 color=TEXT, fontsize=14, y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    periodo_cols = [
        ("Completo",      None,  0),
        ("In-sample",     True,  1),
        ("Out-of-sample", False, 2),
    ]

    for pnombre, in_samp, col in periodo_cols:
        s = resultados[(nombre, pnombre)]
        y, yp, resid = s["y"], s["yp"], s["resid"]

        # ── Fila 0: Fitted vs Real ───────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor(SURFACE)
        ax0.scatter(yp, y, alpha=0.15, s=2, color=color)
        mn, mx = min(yp.min(), y.min()), max(yp.max(), y.max())
        ax0.plot([mn, mx], [mn, mx], color="#facc15", lw=1, ls="--", label="ideal")
        # línea de regresión
        x_line = np.linspace(yp.min(), yp.max(), 200)
        ax0.plot(x_line, s["beta0"] + s["beta1"] * x_line, color=GREEN, lw=1.2, label=f"OLS (β₁={s['beta1']:.4f})")
        ax0.set_title(f"{pnombre}\nR²={s['r2']:.4f}  σ={s['sigma']:.1f}", color=TEXT, fontsize=9)
        ax0.set_xlabel("Predicho", color=MUTED, fontsize=8)
        ax0.set_ylabel("Real", color=MUTED, fontsize=8)
        ax0.tick_params(colors=MUTED, labelsize=7)
        ax0.grid(color=GRID, lw=0.4)
        for sp in ax0.spines.values(): sp.set_color("#222230")
        ax0.legend(fontsize=7, facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")

        # ── Fila 1: Residuos vs Fitted ───────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor(SURFACE)
        ax1.scatter(yp, resid, alpha=0.15, s=2, color=color)
        ax1.axhline(0, color="#facc15", lw=1, ls="--")
        ax1.set_title(f"Residuos vs Fitted\nDW={s['dw']:.3f}  BP_p={s['bp_pval']:.4f}", color=TEXT, fontsize=9)
        ax1.set_xlabel("Predicho", color=MUTED, fontsize=8)
        ax1.set_ylabel("Residuo", color=MUTED, fontsize=8)
        ax1.tick_params(colors=MUTED, labelsize=7)
        ax1.grid(color=GRID, lw=0.4)
        for sp in ax1.spines.values(): sp.set_color("#222230")

        # ── Fila 2: Q-Q plot ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.set_facecolor(SURFACE)
        (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
        ax2.scatter(osm, osr, alpha=0.2, s=2, color=color)
        ax2.plot(osm, slope * np.array(osm) + intercept, color="#facc15", lw=1)
        norm_label = f"{s['norm_test'].split('-')[0]} p={s['norm_pval']:.4f}"
        ax2.set_title(f"Q-Q Plot\n{norm_label}", color=TEXT, fontsize=9)
        ax2.set_xlabel("Cuantiles teóricos", color=MUTED, fontsize=8)
        ax2.set_ylabel("Cuantiles muestrales", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=7)
        ax2.grid(color=GRID, lw=0.4)
        for sp in ax2.spines.values(): sp.set_color("#222230")

    path = PLOTS_DIR / f"regression_diagnostics_{nombre.lower()}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Guardado: {path.name}")


# ── Gráfico comparativo: R² y σ por modelo/periodo ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK_BG)
fig.suptitle("IBEX 35 — Comparativa de Regresión: R², β₁ y σ por Modelo y Periodo",
             color=TEXT, fontsize=13)

periodos_plot = ["Completo", "In-sample", "Out-of-sample"]
modelos_plot  = [n for n, _, _, _ in MODELOS]
colors_plot   = [c for _, _, _, c in MODELOS]
x = np.arange(len(periodos_plot))
w = 0.25

for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.4, axis="y")
    for sp in ax.spines.values(): sp.set_color("#222230")

# R²
for i, (nom, col) in enumerate(zip(modelos_plot, colors_plot)):
    vals = [resultados[(nom, p)]["r2"] for p in periodos_plot]
    axes[0].bar(x + i * w - w, vals, width=w, color=col, alpha=0.85, label=nom)
axes[0].set_title("R²", color=TEXT, fontsize=11)
axes[0].set_xticks(x); axes[0].set_xticklabels(periodos_plot, color=MUTED, fontsize=8)
axes[0].set_ylabel("R²", color=MUTED); axes[0].legend(facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")

# β₁ (pendiente — ideal = 1.0)
for i, (nom, col) in enumerate(zip(modelos_plot, colors_plot)):
    vals = [resultados[(nom, p)]["beta1"] for p in periodos_plot]
    axes[1].bar(x + i * w - w, vals, width=w, color=col, alpha=0.85, label=nom)
axes[1].axhline(1.0, color="#facc15", lw=1, ls="--", label="ideal β₁=1")
axes[1].set_title("β₁ (pendiente, ideal=1.0)", color=TEXT, fontsize=11)
axes[1].set_xticks(x); axes[1].set_xticklabels(periodos_plot, color=MUTED, fontsize=8)
axes[1].set_ylabel("β₁", color=MUTED); axes[1].legend(facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")

# σ (error estándar residuos)
for i, (nom, col) in enumerate(zip(modelos_plot, colors_plot)):
    vals = [resultados[(nom, p)]["sigma"] for p in periodos_plot]
    axes[2].bar(x + i * w - w, vals, width=w, color=col, alpha=0.85, label=nom)
axes[2].set_title("σ (Error estándar residuos)", color=TEXT, fontsize=11)
axes[2].set_xticks(x); axes[2].set_xticklabels(periodos_plot, color=MUTED, fontsize=8)
axes[2].set_ylabel("σ (puntos)", color=MUTED); axes[2].legend(facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")

plt.tight_layout()
path = PLOTS_DIR / "regression_comparison.png"
plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Guardado: {path.name}")

print("\nAnálisis completado.")
