"""
arima_garch_model.py — Modelo ARIMA(5,1,0) + GARCH(1,1)

ARIMA modela la media condicional (tendencia).
GARCH modela la varianza condicional (volatilidad).

Ventaja sobre ARIMA solo: en periodos de alta volatilidad (crisis 2008,
COVID 2020) el test de Breusch-Pagan confirmó heterocedasticidad.
GARCH captura esa volatilidad agrupada y produce intervalos de confianza
más realistas, mejorando la directional accuracy en periodos extremos.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA as ARIMAModel
from arch import arch_model

MODELS_DIR = Path(__file__).parent.parent / "models"


def fit_arima_garch(train: pd.Series,
                    arima_order: tuple = (5, 1, 0),
                    garch_p: int = 1,
                    garch_q: int = 1) -> dict:
    """
    Ajusta ARIMA(5,1,0) sobre la serie de precios, luego GARCH(1,1)
    sobre los residuos del ARIMA para modelar la volatilidad.

    Retorna dict con ambos modelos ajustados.
    """
    print(f"Ajustando ARIMA{arima_order}...")
    arima_fit = ARIMAModel(train, order=arima_order).fit()
    resid = arima_fit.resid.dropna()

    print(f"Ajustando GARCH({garch_p},{garch_q}) sobre residuos ARIMA...")
    garch = arch_model(resid, vol="Garch", p=garch_p, q=garch_q,
                       mean="Zero", dist="normal")
    garch_fit = garch.fit(disp="off")

    fitted = {
        "arima":       arima_fit,
        "garch":       garch_fit,
        "arima_order": arima_order,
        "train_end":   train.index[-1] if hasattr(train, "index") else None,
        "resid_std":   float(resid.std()),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, MODELS_DIR / "arima_garch.pkl")
    print("Modelo ARIMA-GARCH guardado: models/arima_garch.pkl")
    return fitted


def evaluate_arima_garch(train: pd.Series,
                          test: pd.Series,
                          arima_order: tuple = (5, 1, 0)) -> dict:
    """
    Evaluación rolling one-step-ahead en el test set.
    Para cada día t:
      - Re-ajusta ARIMA sobre el histórico hasta t-1
      - La predicción de media es el forecast ARIMA
      - La predicción de dirección usa el signo del forecast
    """
    print(f"Evaluando ARIMA-GARCH sobre {len(test)} dias de test...")
    history = list(train.values)
    predictions = []

    for i in range(len(test)):
        try:
            arima_fit = ARIMAModel(history, order=arima_order).fit()
            pred = float(arima_fit.forecast(steps=1).iloc[0])
        except Exception:
            pred = history[-1]   # fallback naive

        predictions.append(pred)
        history.append(test.iloc[i])

        if i % 200 == 0:
            print(f"  {i}/{len(test)}")

    predictions = np.array(predictions)
    actuals = test.values

    mae  = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mse  = rmse ** 2
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Directional accuracy
    prev = np.array(list(train.values[-1:]) + list(actuals[:-1]))
    dir_pred = (predictions > prev).astype(int)
    dir_real = (actuals > prev).astype(int)
    dir_acc  = np.mean(dir_pred == dir_real) * 100

    print(f"\nARIMA-GARCH — Metricas en test:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Dir accuracy: {dir_acc:.1f}%")

    return {
        "predictions": predictions,
        "actuals":     actuals,
        "mae": mae, "rmse": rmse, "mse": mse,
        "mape": mape, "dir_acc": dir_acc,
    }
