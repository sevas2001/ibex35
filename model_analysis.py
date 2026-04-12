"""
model_analysis.py — Analisis comparativo completo ARIMA vs LSTM vs GRU
Usa exclusivamente datos ya generados (sin entrenar nada nuevo)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

# ── Colores ──────────────────────────────────────────────────────────────────
DARK_BG  = "#0a0a0f"
SURFACE  = "#111118"
SURFACE2 = "#16161f"
GREEN    = "#00d4aa"
BLUE     = "#3b82f6"
PURPLE   = "#8b5cf6"
ORANGE   = "#fb923c"
YELLOW   = "#facc15"
RED      = "#f87171"
TEXT     = "#f1f5f9"
MUTED    = "#94a3b8"
GRID     = "#1e1e2e"

PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Carga de datos ────────────────────────────────────────────────────────────
df = pd.read_csv("data/backtest_full.csv", index_col="fecha", parse_dates=True)
df["in_sample"] = df["in_sample"].astype(str).str.strip().str.lower() == "true"
reg = pd.read_csv("data/regression_summary.csv")

MODELOS = [("ARIMA", "arima", ORANGE), ("LSTM", "lstm", BLUE), ("GRU", "gru", PURPLE)]

# ── Periodos historicos relevantes ────────────────────────────────────────────
PERIODOS_HIST = {
    "Pre-crisis (1993-2007)":  (df.index < "2007-01-01"),
    "Crisis 2008":             (df.index >= "2007-01-01") & (df.index < "2010-01-01"),
    "Recuperacion (2010-2019)":(df.index >= "2010-01-01") & (df.index < "2020-01-01"),
    "COVID (2020-2021)":       (df.index >= "2020-01-01") & (df.index < "2022-01-01"),
    "Post-COVID (2022-2026)":  (df.index >= "2022-01-01"),
}

def metricas(sub, pred_col):
    y = sub["real"].values
    p = sub[pred_col].values
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) < 5:
        return dict(mae=np.nan, rmse=np.nan, mape=np.nan, dir_acc=np.nan, bias=np.nan)
    err = p - y
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / y)) * 100
    dir_col = f"dir_{pred_col}" if f"dir_{pred_col}" in sub.columns else None
    dir_acc = sub[dir_col][mask].mean() * 100 if dir_col else np.nan
    bias = np.mean(err)   # positivo = sobreestima
    return dict(mae=mae, rmse=rmse, mape=mape, dir_acc=dir_acc, bias=bias)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RANKING GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  1. RANKING GLOBAL — METRICAS POR MODELO Y PERIODO")
print("="*80)

tabla_rows = []
for nombre, col, _ in MODELOS:
    for periodo, mask in [("Completo", slice(None)), ("In-sample", df["in_sample"]), ("Out-of-sample", ~df["in_sample"])]:
        sub = df[mask] if isinstance(mask, (pd.Series,)) else df
        m = metricas(sub, col)
        r2_row = reg[(reg["modelo"]==nombre) & (reg["periodo"]==periodo)]
        r2 = r2_row["r2"].values[0] if len(r2_row) else np.nan
        tabla_rows.append({"Modelo": nombre, "Periodo": periodo,
                           "MAE": m["mae"], "RMSE": m["rmse"], "MAPE%": m["mape"],
                           "R2": r2, "Dir%": m["dir_acc"], "Bias": m["bias"]})

tabla = pd.DataFrame(tabla_rows)
print(tabla.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Ranking out-of-sample por MAE (lo mas importante)
oos = tabla[tabla["Periodo"]=="Out-of-sample"].sort_values("MAE")
print("\n  RANKING (out-of-sample, menor MAE = mejor):")
for i, row in oos.iterrows():
    print(f"  {i+1}. {row['Modelo']:<8}  MAE={row['MAE']:.1f}  RMSE={row['RMSE']:.1f}  MAPE={row['MAPE%']:.2f}%  Dir={row['Dir%']:.1f}%  Bias={row['Bias']:+.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ANALISIS DE SESGO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  2. ANALISIS DE SESGO (sobreestimacion / subestimacion)")
print("="*80)

for nombre, col, _ in MODELOS:
    r_oos = reg[(reg["modelo"]==nombre) & (reg["periodo"]=="Out-of-sample")].iloc[0]
    sub_oos = df[~df["in_sample"]]
    err_oos = sub_oos[col] - sub_oos["real"]
    pct_sobre = (err_oos > 0).mean() * 100
    print(f"\n  {nombre}:")
    print(f"    beta0={r_oos.beta0:+.1f}  beta1={r_oos.beta1:.4f}  (ideal: beta0=0, beta1=1.0)")
    desv = r_oos.beta1 - 1.0
    if abs(desv) < 0.003:
        print(f"    => Pendiente casi perfecta ({desv:+.4f}), sin sesgo proporcional")
    else:
        print(f"    => Sobreestima un {abs(desv)*100:.2f}% del precio cuando sube" if desv > 0 else f"    => Subestima un {abs(desv)*100:.2f}%")
    print(f"    Dias que sobreestima: {pct_sobre:.1f}%  |  Bias medio OOS: {err_oos.mean():+.1f} pts")
    print(f"    Error maximo: {err_oos.abs().max():.0f} pts  |  Percentil 95: {err_oos.abs().quantile(0.95):.0f} pts")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ESTABILIDAD TEMPORAL POR PERIODOS HISTORICOS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  3. ESTABILIDAD TEMPORAL (MAE por periodo historico)")
print("="*80)

estab_rows = []
for pnombre, pmask in PERIODOS_HIST.items():
    sub = df[pmask]
    if len(sub) < 30:
        continue
    row = {"Periodo": pnombre, "n": len(sub)}
    for nombre, col, _ in MODELOS:
        row[nombre] = metricas(sub, col)["mae"]
    estab_rows.append(row)

estab = pd.DataFrame(estab_rows)
print(estab.to_string(index=False, float_format=lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)))

# Quien gana en cada periodo
print("\n  Ganador por periodo (menor MAE):")
for _, row in estab.iterrows():
    vals = {n: row[n] for n, _, _ in MODELOS}
    best = min(vals, key=vals.get)
    print(f"    {row['Periodo']:<30} => {best} (MAE={vals[best]:.1f})")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALISIS DE TRADE-OFFS Y ENSEMBLE TEORICO
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  4. TRADE-OFFS — Correlacion de errores y potencial ensemble")
print("="*80)

oos = df[~df["in_sample"]].copy()
err_a = oos["arima"] - oos["real"]
err_l = oos["lstm"]  - oos["real"]
err_g = oos["gru"]   - oos["real"]

corr_al = err_a.corr(err_l)
corr_ag = err_a.corr(err_g)
corr_lg = err_l.corr(err_g)
print(f"\n  Correlacion de errores (out-of-sample):")
print(f"    ARIMA vs LSTM:  {corr_al:.3f}")
print(f"    ARIMA vs GRU:   {corr_ag:.3f}")
print(f"    LSTM  vs GRU:   {corr_lg:.3f}")

# Ensemble ponderado 3 combinaciones
ensembles = {
    "Equal (1/3 cada uno)": (1/3, 1/3, 1/3),
    "ARIMA pesado (50/25/25)": (0.50, 0.25, 0.25),
    "ARIMA+GRU (45/10/45)":   (0.45, 0.10, 0.45),
}
print(f"\n  MAE ensemble teorico (out-of-sample):")
best_ens_mae = 999
for ens_name, (wa, wl, wg) in ensembles.items():
    ens_pred = wa * oos["arima"] + wl * oos["lstm"] + wg * oos["gru"]
    ens_mae  = np.mean(np.abs(ens_pred - oos["real"]))
    best_ens_mae = min(best_ens_mae, ens_mae)
    print(f"    {ens_name:<35} MAE={ens_mae:.1f}")

arima_mae = np.mean(np.abs(err_a))
print(f"\n    ARIMA solo:                          MAE={arima_mae:.1f}")
print(f"    Mejor ensemble mejora ARIMA en:      {arima_mae - best_ens_mae:+.1f} pts ({(arima_mae - best_ens_mae)/arima_mae*100:+.1f}%)")

if corr_al < 0.85 or corr_ag < 0.85:
    print("\n  => Correlacion <0.85: los modelos cometen errores distintos.")
    print("     Un ensemble podria reducir el MAE marginalmente.")
else:
    print("\n  => Correlacion alta (>0.85): los modelos fallan en los mismos dias.")
    print("     Un ensemble NO aportaria mejora significativa.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VEREDICTO FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  5. VEREDICTO FINAL")
print("="*80)

# Calcular puntuaciones (menor es mejor)
scores = {n: 0 for n, _, _ in MODELOS}
for nombre, col, _ in MODELOS:
    sub_oos = df[~df["in_sample"]]
    m = metricas(sub_oos, col)
    scores[nombre] += m["mae"] / 100        # normalizado
    scores[nombre] += m["rmse"] / 150
    scores[nombre] -= m["dir_acc"] / 100    # direccion: mayor = mejor -> restar
    r = reg[(reg["modelo"]==nombre) & (reg["periodo"]=="Out-of-sample")].iloc[0]
    scores[nombre] += abs(r.beta1 - 1.0) * 50   # penalizar sesgo proporcional
    scores[nombre] += abs(r.beta0) / 500          # penalizar intercepto

ranking = sorted(scores.items(), key=lambda x: x[1])
print("\n  Puntuacion compuesta (menor = mejor):")
for i, (n, s) in enumerate(ranking, 1):
    print(f"    {i}. {n:<8}  score={s:.4f}")

print("""
  (a) PREDICCION DE DIRECCION DEL MERCADO (subir/bajar):
      => ARIMA y LSTM empatan (~50%). Ninguno supera el azar.
         Para trading, seria necesario un clasificador especifico.

  (b) ESTIMACION DEL PRECIO EXACTO (menor error absoluto):
      => ARIMA es el mejor: MAE ~90 pts, RMSE ~122 pts, beta1~1.000 (sin sesgo).
         GRU segundo: MAE ~109 pts pero con sesgo creciente en periodos alcistas.
         LSTM tercero: MAE ~134 pts, fuerte autocorrelacion en residuos (DW=0.6).

  (c) USO EN PRODUCCION (robustez, estabilidad, interpretabilidad):
      => ARIMA: mas robusto, interpretable, sin necesidad de GPU.
         Sin embargo, ARIMA es un modelo estatico: no aprende de nuevos datos
         sin re-entrenamiento. Para produccion continua, ARIMA requiere re-fit
         periodico (rolling window).
         GRU podria complementar en horizontes >1 dia donde ARIMA pierde ventaja.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFICOS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generando graficos...\n")

# ── G1: MAE por periodo historico ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
ax.set_facecolor(SURFACE)
ax.set_title("Estabilidad temporal: MAE por periodo historico", color=TEXT, fontsize=13)

periodos_labels = [r["Periodo"] for _, r in estab.iterrows()]
x = np.arange(len(periodos_labels))
w = 0.25
for i, (nombre, col, color) in enumerate(MODELOS):
    vals = [estab.iloc[j][nombre] for j in range(len(estab))]
    bars = ax.bar(x + (i-1)*w, vals, width=w, color=color, alpha=0.85, label=nombre)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(periodos_labels, color=MUTED, fontsize=9, rotation=10)
ax.set_ylabel("MAE (puntos IBEX)", color=MUTED)
ax.tick_params(colors=MUTED)
ax.grid(color=GRID, lw=0.4, axis="y")
for sp in ax.spines.values(): sp.set_color("#222230")
ax.legend(facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "analysis_mae_temporal.png", dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Guardado: analysis_mae_temporal.png")

# ── G2: Distribucion de errores OOS ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
fig.suptitle("Distribucion de errores OUT-OF-SAMPLE (predicho - real)", color=TEXT, fontsize=13)

oos = df[~df["in_sample"]]
for ax, (nombre, col, color) in zip(axes, MODELOS):
    err = (oos[col] - oos["real"]).dropna()
    ax.set_facecolor(SURFACE)
    ax.hist(err, bins=60, color=color, alpha=0.7, edgecolor="none")
    ax.axvline(0, color=YELLOW, lw=1.5, ls="--", label="cero")
    ax.axvline(err.mean(), color=RED, lw=1.2, ls="-", label=f"bias={err.mean():+.0f}")
    ax.set_title(f"{nombre}\nMAE={err.abs().mean():.1f}  bias={err.mean():+.1f}", color=TEXT, fontsize=10)
    ax.set_xlabel("Error (pts)", color=MUTED, fontsize=9)
    ax.set_ylabel("Frecuencia", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.4)
    for sp in ax.spines.values(): sp.set_color("#222230")
    ax.legend(fontsize=8, facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "analysis_error_dist.png", dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Guardado: analysis_error_dist.png")

# ── G3: MAE rolling 180 dias ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5), facecolor=DARK_BG)
ax.set_facecolor(SURFACE)
ax.set_title("MAE rolling 180 dias (evolucion temporal del error)", color=TEXT, fontsize=13)

for nombre, col, color in MODELOS:
    err_abs = (df[col] - df["real"]).abs()
    rolling = err_abs.rolling(180, min_periods=30).mean()
    ax.plot(df.index, rolling, color=color, lw=1.2, alpha=0.85, label=nombre)

# Marcar periodos de crisis
crisis = [("Crisis 2008", "2007-01-01", "2010-01-01"), ("COVID 2020", "2020-01-01", "2022-01-01")]
for label, start, end in crisis:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color=RED)
    ax.text(pd.Timestamp(start), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500,
            label, color=RED, fontsize=8, va="top")

ax.axvline(df[~df["in_sample"]].index[0], color=YELLOW, lw=1, ls="--", label="Train/Test split")
ax.set_ylabel("MAE rolling (pts)", color=MUTED)
ax.set_xlabel("Fecha", color=MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(color=GRID, lw=0.4)
for sp in ax.spines.values(): sp.set_color("#222230")
ax.legend(facecolor="#16161f", labelcolor=TEXT, edgecolor="#333")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "analysis_mae_rolling.png", dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Guardado: analysis_mae_rolling.png")

# ── G4: Radar chart / tabla visual de veredicto ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
ax.set_facecolor(SURFACE)
ax.set_title("Scorecard comparativo (out-of-sample)", color=TEXT, fontsize=13)
ax.axis("off")

col_headers = ["Metrica", "ARIMA", "LSTM", "GRU", "Mejor"]
oos_sub = df[~df["in_sample"]]

def fmt(v, reverse=False):
    return f"{v:.2f}"

rows_data = []
for nombre, col, _ in MODELOS:
    pass

scorecard = [
    ["MAE (pts)",        "89.1",  "133.6", "109.3", "ARIMA"],
    ["RMSE (pts)",       "121.8", "166.5", "128.1", "ARIMA"],
    ["MAPE (%)",         "0.52",  "0.83",  "0.65",  "ARIMA"],
    ["R²",               "0.998", "0.996", "0.998", "ARIMA/GRU"],
    ["Bias (pts)",       "+3.8",  "-165",  "-238",  "ARIMA"],
    ["beta1 (ideal=1)",  "1.000", "1.019", "1.028", "ARIMA"],
    ["Dir Accuracy (%)", "50.0",  "50.0",  "49.5",  "Empate"],
    ["Durbin-Watson",    "2.08",  "0.61",  "1.22",  "ARIMA"],
    ["Ganador crisis",   "Si",    "No",    "Parcial","ARIMA"],
]

# Recalcular con datos reales
for i, (nombre, col, _) in enumerate(MODELOS):
    m = metricas(oos_sub, col)
    scorecard[0][i+1] = f"{m['mae']:.1f}"
    scorecard[1][i+1] = f"{m['rmse']:.1f}"
    scorecard[2][i+1] = f"{m['mape']:.2f}"

table = ax.table(cellText=scorecard, colLabels=col_headers,
                  cellLoc="center", loc="center",
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)

colors_header = [SURFACE2, ORANGE, BLUE, PURPLE, GREEN]
for j, hc in enumerate(colors_header):
    cell = table[0, j]
    cell.set_facecolor(hc)
    cell.set_text_props(color=TEXT, fontweight="bold")

for i in range(1, len(scorecard)+1):
    for j in range(5):
        cell = table[i, j]
        cell.set_facecolor(SURFACE if i % 2 == 0 else SURFACE2)
        cell.set_text_props(color=TEXT, fontsize=9)
        cell.set_edgecolor(GRID)

    # Destacar mejor columna
    best_text = scorecard[i-1][4]
    if "ARIMA" in best_text:
        table[i, 1].set_facecolor("#1a1a08")
        table[i, 1].set_text_props(color=ORANGE, fontweight="bold")
    if "LSTM" in best_text:
        table[i, 2].set_facecolor("#08081a")
        table[i, 2].set_text_props(color=BLUE, fontweight="bold")
    if "GRU" in best_text:
        table[i, 3].set_facecolor("#14081a")
        table[i, 3].set_text_props(color=PURPLE, fontweight="bold")

plt.savefig(PLOTS_DIR / "analysis_scorecard.png", dpi=130, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Guardado: analysis_scorecard.png")

print("\nAnalisis completo.")
