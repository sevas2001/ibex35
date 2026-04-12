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
    "prediccion_d1",      # predicción LSTM para el día hábil siguiente
    "prediccion_arima",   # predicción ARIMA para el día hábil siguiente
    "real_d1",            # precio real del día siguiente (se rellena al día siguiente)
    "error_abs",          # |real - prediccion_lstm|
    "error_arima",        # |real - prediccion_arima|
    "error_pct",          # error en %
    "direction_correct",  # 1 si acertó dirección (LSTM), 0 si no, None si aún no hay real
    "direction_arima",    # 1 si ARIMA acertó dirección
]


def _init_log() -> pd.DataFrame:
    """Crea el CSV de log si no existe."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(LOG_PATH, index=False)
    return pd.read_csv(LOG_PATH)


def save_prediction(precio_base: float, prediccion_d1: float,
                    prediccion_arima: float = None,
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
        "prediccion_arima": round(prediccion_arima, 2) if prediccion_arima is not None else None,
        "real_d1": None,
        "error_abs": None,
        "error_arima": None,
        "error_pct": None,
        "direction_correct": None,
        "direction_arima": None,
    }])
    df = pd.concat([df, nueva], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"Prediccion guardada: {hoy} -> {prediccion_d1:.2f} pts (base: {precio_base:.2f})")


def _load_close_map(min_date: str) -> dict:
    """
    Construye un mapa fecha->precio intentando yfinance primero,
    y usando el CSV histórico bundleado como fallback.
    """
    # Intentar yfinance
    for attempt in range(3):
        try:
            recent = yf.download("^IBEX", start=min_date,
                                 auto_adjust=True, progress=False)
            if isinstance(recent.columns, pd.MultiIndex):
                recent.columns = recent.columns.get_level_values(0)
            if not recent.empty:
                return {d.strftime("%Y-%m-%d"): float(p)
                        for d, p in zip(recent.index, recent["Close"])}
        except Exception:
            pass
        if attempt < 2:
            import time
            time.sleep(2)

    # Fallback: CSV histórico
    raw_path = LOG_PATH.parent / "raw" / "ibex35_raw.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        return {d.strftime("%Y-%m-%d"): float(p)
                for d, p in zip(raw.index, raw["Close"])}
    return {}


def update_with_real_prices() -> pd.DataFrame:
    """
    Rellena precios reales en filas pendientes del log.
    Usa yfinance con fallback al CSV histórico bundleado.
    """
    df = _init_log()
    if df.empty:
        return df

    # Filas sin precio real
    pending = df[df["real_d1"].isna() & df["fecha_prediccion"].notna()].copy()
    if pending.empty:
        return df

    min_date = pending["fecha_prediccion"].min()
    close_map = _load_close_map(min_date)
    if not close_map:
        return df

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

            # ARIMA si existe
            arima_col = "prediccion_arima"
            if arima_col in df.columns and pd.notna(row.get(arima_col)):
                pred_a = float(row[arima_col])
                err_a  = abs(real - pred_a)
                dir_a  = int((real > base) == (pred_a > base))
                df.at[idx, "error_arima"]    = round(err_a, 2)
                df.at[idx, "direction_arima"] = dir_a

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
            "direction_arima_pct": None,
            "mae_arima": None,
            "historial": [],
        }

    evaluated["error_abs"] = evaluated["error_abs"].astype(float)
    evaluated["direction_correct"] = evaluated["direction_correct"].astype(float)

    direction_acc = round(evaluated["direction_correct"].mean() * 100, 1)
    mae = round(evaluated["error_abs"].mean(), 2)
    rmse = round(np.sqrt((evaluated["error_abs"] ** 2).mean()), 2)

    # Métricas ARIMA (solo filas donde existe prediccion_arima)
    direction_arima_pct = None
    mae_arima = None
    if "error_arima" in evaluated.columns:
        arima_eval = evaluated[evaluated["error_arima"].notna()].copy()
        if not arima_eval.empty:
            arima_eval["error_arima"] = arima_eval["error_arima"].astype(float)
            mae_arima = round(arima_eval["error_arima"].mean(), 2)
            if "direction_arima" in arima_eval.columns:
                arima_eval["direction_arima"] = arima_eval["direction_arima"].astype(float)
                direction_arima_pct = round(arima_eval["direction_arima"].mean() * 100, 1)

    historial = evaluated.sort_values("fecha_prediccion", ascending=False).head(30)
    historial = historial.fillna("").to_dict(orient="records")

    return {
        "total_predicciones": len(df),
        "evaluadas": len(evaluated),
        "direction_accuracy_pct": direction_acc,
        "mae": mae,
        "rmse": rmse,
        "direction_arima_pct": direction_arima_pct,
        "mae_arima": mae_arima,
        "historial": historial,
    }
