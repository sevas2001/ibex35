import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

RAW_DATA_PATH  = Path(__file__).parent.parent / "data" / "raw" / "ibex35_raw.csv"
FULL_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "ibex35_full.csv"

# Fecha más antigua con datos fiables en yfinance para ^IBEX
EARLIEST_DATE = "1993-01-01"


def download_ibex35(start_date: str = EARLIEST_DATE, force: bool = False) -> pd.DataFrame:
    """
    Descarga datos históricos del IBEX 35 desde Yahoo Finance.
    Retorna DataFrame con columnas: Open, High, Low, Close, Volume.
    Por defecto descarga desde 1993 (máximo disponible en yfinance).
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Descargando IBEX 35 desde {start_date} hasta {end_date}...")

    try:
        data = yf.download("^IBEX", start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            raise ValueError("yfinance devolvio un DataFrame vacio.")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index.name = "Date"
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

        # Guardar en ambas rutas para compatibilidad
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(RAW_DATA_PATH)
        data.to_csv(FULL_DATA_PATH)

        print(f"Total registros: {len(data)} | Desde: {data.index[0].date()} hasta: {data.index[-1].date()}")
        print(f"Guardado en {FULL_DATA_PATH}")
        return data

    except Exception as e:
        raise RuntimeError(f"Error al descargar datos: {e}")


def load_ibex35(full_history: bool = True) -> pd.DataFrame:
    """
    Carga datos del IBEX 35.
    full_history=True  → usa ibex35_full.csv (desde 1993)
    full_history=False → usa ibex35_raw.csv  (compatibilidad)
    Si el CSV no existe o tiene datos solo desde 2012, re-descarga.
    """
    path = FULL_DATA_PATH if full_history else RAW_DATA_PATH

    if path.exists():
        data = pd.read_csv(path, index_col="Date", parse_dates=True)
        first_year = data.index[0].year
        # Si solo tenemos datos desde 2012, re-descargamos el histórico completo
        if full_history and first_year > 1999:
            print(f"CSV existente solo desde {first_year} — re-descargando historico completo...")
            return download_ibex35(start_date=EARLIEST_DATE)
        print(f"Cargando datos desde {path} ({len(data)} registros, desde {data.index[0].date()})")
        return data

    return download_ibex35(start_date=EARLIEST_DATE if full_history else "2012-01-01")
