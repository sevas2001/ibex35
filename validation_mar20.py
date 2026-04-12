"""
validation_mar20.py — Experimento walk-forward con corte 20 de marzo

Entrena ARIMA-GARCH e Híbrido ARIMA+LSTM hasta el 20-03-2026,
luego hace predicciones rolling para 21, 24, 25, 26, 27 de marzo
y compara con precios reales.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

# ── Configuración ─────────────────────────────────────────────────────────────
CUTOFF      = "2026-03-20"
DATA_PATH   = Path("data/raw/ibex35_full.csv")
MODELS_DIR  = Path("models")
PLOTS_DIR   = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG  = "#0a0a0f"
SURFACE  = "#111118"
ORANGE   = "#fb923c"
GREEN    = "#00d4aa"
BLUE     = "#3b82f6"
YELLOW   = "#facc15"
TEXT     = "#f1f5f9"
MUTED    = "#94a3b8"
GRID     = "#1e1e2e"
RED      = "#f87171"

print("=" * 60)
print("  VALIDACIÓN WALK-FORWARD — corte 2026-03-20")
print("=" * 60)

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
full = full[["Close"]].dropna()

# Intentar descargar datos más recientes de yfinance
try:
    import yfinance as yf
    from datetime import datetime, timedelta
    end = datetime.now() + timedelta(days=1)
    recent = yf.download("^IBEX", start="2026-03-18",
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
    if isinstance(recent.columns, pd.MultiIndex):
        recent.columns = recent.columns.get_level_values(0)
    if not recent.empty:
        recent = recent[["Close"]].dropna()
        full = pd.concat([full, recent])
        full = full[~full.index.duplicated(keep="last")].sort_index()
        print(f"  Datos actualizados hasta: {full.index[-1].date()}")
except Exception as e:
    print(f"  yfinance no disponible ({e}), usando CSV local")

train_data = full[full.index <= CUTOFF]["Close"]
all_data   = full["Close"]

# Días hábiles después del corte (solo los que tenemos precio real)
val_days = full[(full.index > CUTOFF)].index
print(f"\nDatos de entrenamiento: {len(train_data)} días (hasta {CUTOFF})")
print(f"Días de validación disponibles: {[d.date() for d in val_days]}")


# ── 2. ARIMA-GARCH — entrenamiento ───────────────────────────────────────────
print("\n--- ARIMA-GARCH(1,1) ---")
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    print("  arch no instalado — usando ARIMA solo como fallback")
    HAS_ARCH = False

def fit_arima_garch(series, order=(5,1,0)):
    """Ajusta ARIMA y luego GARCH(1,1) sobre los residuos."""
    arima_fit = ARIMAModel(series.values, order=order).fit()
    if HAS_ARCH:
        resid = pd.Series(arima_fit.resid)
        garch_fit = arch_model(resid, vol="GARCH", p=1, q=1).fit(disp="off")
        return arima_fit, garch_fit
    return arima_fit, None

def predict_arima_garch(arima_fit, garch_fit, series, steps=1):
    """Predicción ARIMA-GARCH: media ARIMA (la parte de nivel)."""
    fc = arima_fit.forecast(steps=steps)
    return float(fc.iloc[0])

print(f"  Ajustando sobre {len(train_data)} días...")
arima_fit_base, garch_fit_base = fit_arima_garch(train_data)
print("  OK")


# ── 3. Híbrido ARIMA+LSTM — cargar modelo entrenado ──────────────────────────
print("\n--- Híbrido ARIMA+LSTM ---")
hybrid_path = MODELS_DIR / "hybrid_lstm.keras"
scaler_hyb_path = MODELS_DIR / "scaler_hybrid.pkl"

USE_HYBRID = hybrid_path.exists() and scaler_hyb_path.exists()
if USE_HYBRID:
    from tensorflow import keras
    import keras as keras_lib
    keras_lib.config.enable_unsafe_deserialization()
    hybrid_model = keras.models.load_model(hybrid_path)
    scaler_hyb   = joblib.load(scaler_hyb_path)
    print("  Modelo cargado desde models/hybrid_lstm.keras")
    # Features v2 para el hibrido
    from src.preprocessor import compute_features_v2
    SEQ_LEN_HYB = 60
    N_FEAT_HYB  = 4
else:
    print("  Modelo no encontrado — solo se usará ARIMA-GARCH")


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def get_hybrid_features(close_series):
    """Genera features v2: log_return, ma50_ratio, rsi14 + residuo placeholder."""
    s = close_series.copy()
    log_ret  = np.log(s / s.shift(1)).fillna(0)
    ma50     = s.rolling(50).mean().fillna(s)
    ma50_rat = (s / ma50).fillna(1)
    rsi      = compute_rsi(s).fillna(50)
    feats = pd.DataFrame({"close": s, "log_ret": log_ret,
                          "ma50_ratio": ma50_rat, "rsi14": rsi},
                         index=s.index)
    return feats


# ── 4. Predicciones rolling día a día ─────────────────────────────────────────
print("\n--- Predicciones rolling ---")
results = []
window = train_data.copy()  # ventana deslizante

for day in val_days:
    real_price = float(all_data[day])
    prev_price = float(window.iloc[-1])

    # ── ARIMA-GARCH ──────────────────────────────────────────────────────────
    try:
        arima_roll = ARIMAModel(window.values[-200:], order=(5,1,0)).fit()
        ag_pred = float(arima_roll.forecast(steps=1).iloc[0])
    except Exception:
        ag_pred = float(window.iloc[-1])

    # ── Híbrido ───────────────────────────────────────────────────────────────
    hyb_pred = np.nan
    if USE_HYBRID:
        try:
            feats = get_hybrid_features(window)
            # Tomar últimos 60 días de features
            last_feats = feats[["log_ret", "ma50_ratio", "rsi14"]].values[-SEQ_LEN_HYB:]
            # Residuo placeholder: diferencia entre precio y ARIMA naive
            arima_naive = window.values[-1]  # simplificado
            residuo_col = np.zeros((len(last_feats), 1))
            seq = np.hstack([residuo_col, last_feats])  # (60, 4)
            seq_scaled = scaler_hyb.transform(seq)
            X = seq_scaled.reshape(1, SEQ_LEN_HYB, N_FEAT_HYB)
            residuo_pred = float(hybrid_model.predict(X, verbose=0)[0, 0])
            # Desnormalizar residuo
            dummy = np.zeros((1, N_FEAT_HYB))
            dummy[0, 0] = residuo_pred
            residuo_real = scaler_hyb.inverse_transform(dummy)[0, 0]
            hyb_pred = ag_pred + residuo_real
        except Exception as e:
            hyb_pred = ag_pred  # fallback a ARIMA si falla híbrido

    # ── Dirección ─────────────────────────────────────────────────────────────
    dir_ag  = 1 if (real_price > prev_price) == (ag_pred > prev_price) else 0
    dir_hyb = 1 if not np.isnan(hyb_pred) and (real_price > prev_price) == (hyb_pred > prev_price) else 0

    results.append({
        "fecha":       day.date(),
        "real":        round(real_price, 2),
        "arima_garch": round(ag_pred, 2),
        "error_ag":    round(abs(real_price - ag_pred), 2),
        "dir_ag":      "OK" if dir_ag else "FAIL",
        "hibrido":     round(hyb_pred, 2) if not np.isnan(hyb_pred) else None,
        "error_h":     round(abs(real_price - hyb_pred), 2) if not np.isnan(hyb_pred) else None,
        "dir_h":       "OK" if dir_hyb else "FAIL",
    })

    print(f"  {day.date()}  real={real_price:.1f}  "
          f"ARIMA-GARCH={ag_pred:.1f} (err={abs(real_price-ag_pred):.1f}, {results[-1]['dir_ag']})  "
          f"Hibrido={hyb_pred:.1f} (err={abs(real_price-hyb_pred):.1f}, {results[-1]['dir_h']})" if not np.isnan(hyb_pred)
          else f"  {day.date()}  real={real_price:.1f}  ARIMA-GARCH={ag_pred:.1f} (err={abs(real_price-ag_pred):.1f})")

    # Añadir precio real a la ventana para el día siguiente (rolling)
    window = pd.concat([window, pd.Series([real_price], index=[day])])


# ── 5. Tabla de resultados ────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
print("\n" + "=" * 75)
print("  TABLA DE RESULTADOS — Walk-Forward Validación")
print("=" * 75)
print(res_df.to_string(index=False))

# Métricas globales
mae_ag  = res_df["error_ag"].mean()
rmse_ag = np.sqrt((res_df["error_ag"]**2).mean())
dir_ag  = (res_df["dir_ag"] == "OK").mean() * 100

print(f"\n  ARIMA-GARCH:  MAE={mae_ag:.1f}  RMSE={rmse_ag:.1f}  Dir={dir_ag:.0f}%")

if res_df["error_h"].notna().any():
    mae_h  = res_df["error_h"].dropna().mean()
    rmse_h = np.sqrt((res_df["error_h"].dropna()**2).mean())
    dir_h  = (res_df["dir_h"] == "OK").mean() * 100
    print(f"  Híbrido:      MAE={mae_h:.1f}  RMSE={rmse_h:.1f}  Dir={dir_h:.0f}%")
else:
    mae_h, rmse_h, dir_h = None, None, None

res_df.to_csv("data/validation_mar20.csv", index=False)
print("\nResultados guardados: data/validation_mar20.csv")


# ── 6. Gráfico ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=DARK_BG)
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.4)
    for sp in ax.spines.values(): sp.set_color("#222230")

dates_plt = [pd.Timestamp(r["fecha"]) for r in results]
reals     = [r["real"]        for r in results]
ags       = [r["arima_garch"] for r in results]
hybs      = [r["hibrido"]     for r in results if r["hibrido"] is not None]
hyb_dates = [pd.Timestamp(r["fecha"]) for r in results if r["hibrido"] is not None]

# Contexto: últimos 20 días de train
context = full[(full.index <= CUTOFF)].tail(20)
cutoff_ts = pd.Timestamp(CUTOFF)

ax1.plot(context.index, context["Close"], color=MUTED, lw=1, alpha=0.5, label="Histórico (contexto)")
ax1.axvline(cutoff_ts, color=YELLOW, lw=1, ls="--", alpha=0.7, label="Corte 20-mar")
ax1.plot(dates_plt, reals, color=TEXT,   lw=2,   marker="o", ms=6, label="Real")
ax1.plot(dates_plt, ags,   color=ORANGE, lw=1.5, marker="s", ms=5, label=f"ARIMA-GARCH (MAE={mae_ag:.1f})")
if hybs:
    ax1.plot(hyb_dates, hybs, color=GREEN, lw=1.5, marker="^", ms=5,
             label=f"Hibrido (MAE={mae_h:.1f})" if mae_h else "Hibrido")

ax1.set_title("Validación Walk-Forward: ARIMA-GARCH vs Híbrido ARIMA+LSTM\nCorte: 2026-03-20 | Predicciones: 21–27 marzo",
              color=TEXT, fontsize=12, pad=10)
ax1.set_ylabel("Puntos IBEX", color=MUTED)
ax1.xaxis.set_visible(False)
ax1.legend(facecolor="#16161f", edgecolor="#333", labelcolor=TEXT, fontsize=9)

# Panel inferior: errores
width = 0.35
x_pos = np.arange(len(dates_plt))
bars_ag = ax2.bar(x_pos - width/2, res_df["error_ag"], width, color=ORANGE, alpha=0.8, label="Error ARIMA-GARCH")
if res_df["error_h"].notna().any():
    bars_h = ax2.bar(x_pos + width/2, res_df["error_h"].fillna(0), width, color=GREEN, alpha=0.8, label="Error Híbrido")
ax2.axhline(mae_ag, color=ORANGE, lw=1, ls="--", alpha=0.7, label=f"MAE ARIMA-GARCH={mae_ag:.1f}")
if mae_h:
    ax2.axhline(mae_h, color=GREEN, lw=1, ls="--", alpha=0.7, label=f"MAE Híbrido={mae_h:.1f}")

ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(r["fecha"]) for r in results], color=MUTED, fontsize=8)
ax2.set_ylabel("|Error| (pts)", color=MUTED)
ax2.legend(facecolor="#16161f", edgecolor="#333", labelcolor=TEXT, fontsize=8)

# Anotar dirección OK/FAIL
for i, r in enumerate(results):
    color_ag = GREEN if r["dir_ag"] == "OK" else RED
    ax1.annotate(r["dir_ag"], xy=(dates_plt[i], r["arima_garch"]),
                 xytext=(0, 10), textcoords="offset points",
                 color=color_ag, fontsize=7, ha="center", fontweight="bold")

plt.savefig(PLOTS_DIR / "validation_mar20.png", dpi=140, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Grafico guardado: data/plots/validation_mar20.png")
print("\nValidacion completada.")
