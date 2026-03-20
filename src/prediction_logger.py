"""
Sistema de logging de predicciones.
Guarda la predicción de hoy y al día siguiente compara con el precio real.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "data" / "prediction_log.csv"

COLUMNS = [
    "fecha_prediccion",   # fecha en que se hizo la predicción
    "precio_base",        # último precio real en el momento de predecir
    "prediccion_d1",      # predicción para el día hábil siguiente
    "real_d1",            # precio real del día siguiente (se rellena al día siguiente)
    "error_abs",          # |real - prediccion|
    "error_pct",          # error en %
    "direction_correct",  # 1 si acertó dirección, 0 si no, None si aún no hay real
]


def _init_log() -> pd.DataFrame:
    """Crea el CSV de log si no existe."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(LOG_PATH, index=False)
    return pd.read_csv(LOG_PATH)


def save_prediction(precio_base: float, prediccion_d1: float,
                    fecha: str = None) -> None:
    """
    Guarda la predicción de hoy en el log.
    fecha: 'YYYY-MM-DD', por defecto hoy.
    """
    df = _init_log()
    hoy = fecha or date.today().isoformat()

    # Evitar duplicados para el mismo día
    if hoy in df["fecha_prediccion"].values:
        print(f"Ya existe predicción para {hoy}, actualizando...")
        df = df[df["fecha_prediccion"] != hoy]

    nueva = pd.DataFrame([{
        "fecha_prediccion": hoy,
        "precio_base": round(precio_base, 2),
        "prediccion_d1": round(prediccion_d1, 2),
        "real_d1": None,
        "error_abs": None,
        "error_pct": None,
        "direction_correct": None,
    }])
    df = pd.concat([df, nueva], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"Prediccion guardada: {hoy} -> {prediccion_d1:.2f} pts (base: {precio_base:.2f})")


def update_with_real_prices() -> pd.DataFrame:
    """
    Descarga precios reales de yfinance y rellena las filas pendientes.
    Se llama automáticamente cada vez que se carga el log.
    """
    df = _init_log()
    if df.empty:
        return df

    # Filas sin precio real
    pending = df[df["real_d1"].isna() & df["fecha_prediccion"].notna()].copy()
    if pending.empty:
        return df

    # Descargar datos recientes
    min_date = pending["fecha_prediccion"].min()
    recent = yf.download("^IBEX", start=min_date, auto_adjust=True, progress=False)
    if isinstance(recent.columns, pd.MultiIndex):
        recent.columns = recent.columns.get_level_values(0)
    if recent.empty:
        return df

    close_map = {d.strftime("%Y-%m-%d"): float(p)
                 for d, p in zip(recent.index, recent["Close"])}

    for idx, row in pending.iterrows():
        fecha = row["fecha_prediccion"]
        # Buscamos el siguiente día hábil después de la predicción
        future = pd.bdate_range(start=fecha, periods=2)[1]
        future_str = future.strftime("%Y-%m-%d")

        if future_str in close_map:
            real = close_map[future_str]
            pred = float(row["prediccion_d1"])
            base = float(row["precio_base"])
            error = abs(real - pred)
            error_pct = round(error / real * 100, 2)
            direction_correct = int((real > base) == (pred > base))

            df.at[idx, "real_d1"] = round(real, 2)
            df.at[idx, "error_abs"] = round(error, 2)
            df.at[idx, "error_pct"] = error_pct
            df.at[idx, "direction_correct"] = direction_correct

    df.to_csv(LOG_PATH, index=False)
    return df


def get_accuracy_summary() -> dict:
    """Calcula métricas de accuracy del log completo."""
    df = update_with_real_prices()
    evaluated = df[df["real_d1"].notna()].copy()

    if evaluated.empty:
        return {
            "total_predicciones": 0,
            "evaluadas": 0,
            "direction_accuracy_pct": None,
            "mae": None,
            "rmse": None,
            "historial": [],
        }

    evaluated["error_abs"] = evaluated["error_abs"].astype(float)
    evaluated["direction_correct"] = evaluated["direction_correct"].astype(float)

    direction_acc = round(evaluated["direction_correct"].mean() * 100, 1)
    mae = round(evaluated["error_abs"].mean(), 2)
    rmse = round(np.sqrt((evaluated["error_abs"] ** 2).mean()), 2)

    historial = evaluated.sort_values("fecha_prediccion", ascending=False).head(30)
    historial = historial.fillna("").to_dict(orient="records")

    return {
        "total_predicciones": len(df),
        "evaluadas": len(evaluated),
        "direction_accuracy_pct": direction_acc,
        "mae": mae,
        "rmse": rmse,
        "historial": historial,
    }
