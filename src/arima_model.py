import numpy as np
import pandas as pd
import warnings
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent.parent / "models"
PLOTS_DIR = Path(__file__).parent.parent / "data" / "plots"


def test_stationarity(series: pd.Series) -> bool:
    """Test ADF para verificar estacionariedad."""
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"ADF Test — p-value: {p_value:.4f} ({'estacionaria' if p_value < 0.05 else 'NO estacionaria'})")
    return p_value < 0.05


def plot_acf_pacf(series: pd.Series) -> None:
    """Grafica ACF y PACF para seleccionar parámetros ARIMA."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    diff_series = series.diff().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    plot_acf(series, lags=40, ax=axes[0, 0], title="ACF — Serie original")
    plot_pacf(series, lags=40, ax=axes[0, 1], title="PACF — Serie original")
    plot_acf(diff_series, lags=40, ax=axes[1, 0], title="ACF — Serie diferenciada (d=1)")
    plot_pacf(diff_series, lags=40, ax=axes[1, 1], title="PACF — Serie diferenciada (d=1)")
    plt.tight_layout()
    path = PLOTS_DIR / "acf_pacf.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ACF/PACF guardado: {path}")


def train_arima(train: pd.Series, order: tuple = (5, 1, 0)) -> object:
    """
    Entrena modelo ARIMA con los parámetros dados.
    Orden por defecto (5,1,0) basado en análisis ACF/PACF típico del IBEX.
    """
    print(f"Entrenando ARIMA{order}...")
    model = ARIMA(train, order=order)
    fitted = model.fit()
    print(fitted.summary().tables[0])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, MODELS_DIR / "arima_model.pkl")
    print("Modelo ARIMA guardado en models/arima_model.pkl")
    return fitted


def predict_arima(fitted_model, n_steps: int, last_train: pd.Series) -> np.ndarray:
    """Genera predicciones con el modelo ARIMA."""
    forecast = fitted_model.forecast(steps=n_steps)
    return np.array(forecast)


def evaluate_arima(fitted_model, train: pd.Series, test: pd.Series) -> dict:
    """Genera predicciones sobre el test set y calcula métricas."""
    print(f"Generando predicciones ARIMA sobre {len(test)} días de test...")

    # Predicción rolling: reentrenar con cada nuevo dato real
    predictions = []
    history = list(train)

    for i in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        fitted = model.fit()
        pred = fitted.forecast(steps=1)[0]
        predictions.append(pred)
        history.append(test.iloc[i])

        if i % 100 == 0:
            print(f"  Progreso: {i}/{len(test)}")

    predictions = np.array(predictions)
    actuals = test.values

    mse = np.mean((actuals - predictions) ** 2)
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(mse)

    print(f"\nARIMA — Metricas en test:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    return {"predictions": predictions, "mse": mse, "mae": mae, "rmse": rmse}
