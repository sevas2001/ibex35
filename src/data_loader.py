import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "ibex35_raw.csv"


def download_ibex35(start_date: str = "2012-01-01") -> pd.DataFrame:
    """
    Descarga datos históricos del IBEX 35 desde Yahoo Finance.
    Retorna DataFrame con columnas: Open, High, Low, Close, Volume.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Descargando IBEX 35 desde {start_date} hasta {end_date}...")

    try:
        data = yf.download("^IBEX", start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            raise ValueError("yfinance devolvió un DataFrame vacío.")

        # Aplanar columnas multinivel si existen
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index.name = "Date"
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(RAW_DATA_PATH)
        print(f"Datos guardados en {RAW_DATA_PATH}")
        print(f"Total registros: {len(data)} | Desde: {data.index[0].date()} hasta: {data.index[-1].date()}")
        return data

    except Exception as e:
        raise RuntimeError(f"Error al descargar datos: {e}")


def load_ibex35() -> pd.DataFrame:
    """Carga los datos desde CSV si ya existen, si no los descarga."""
    if RAW_DATA_PATH.exists():
        print(f"Cargando datos desde {RAW_DATA_PATH}")
        data = pd.read_csv(RAW_DATA_PATH, index_col="Date", parse_dates=True)
        return data
    return download_ibex35()
