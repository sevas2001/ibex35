"""
Walk-Forward Validation para LSTM.
Reentrena el modelo progresivamente añadiendo cada día nuevo al train set.
Fecha límite: 31 de marzo de 2026.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = DATA_DIR / "plots"
SEQUENCE_LENGTH = 60
FINETUNE_EPOCHS = 5   # epochs por cada nuevo día (balance velocidad/precisión)


def _load_model_and_scaler():
    from tensorflow import keras
    model = keras.models.load_model(MODELS_DIR / "lstm_model.keras")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


def _predict_one_step(model, scaler, last_60_real: np.ndarray) -> float:
    """Predice el siguiente precio dado los últimos 60 días reales."""
    scaled = scaler.transform(last_60_real.reshape(-1, 1))
    X = scaled.reshape(1, SEQUENCE_LENGTH, 1)
    pred_scaled = float(model.predict(X, verbose=0)[0, 0])
    pred_real = float(scaler.inverse_transform([[pred_scaled]])[0, 0])
    return pred_real


def _finetune(model, scaler, recent_prices: np.ndarray) -> None:
    """Fine-tune el modelo con los datos más recientes (últimos 60+1 días)."""
    if len(recent_prices) < SEQUENCE_LENGTH + 1:
        return

    scaled = scaler.transform(recent_prices.reshape(-1, 1))
    X = scaled[:-1].reshape(1, SEQUENCE_LENGTH, 1)
    y = scaled[-1].reshape(1, 1)

    model.fit(X, y, epochs=FINETUNE_EPOCHS, verbose=0)


def run_walk_forward(data: pd.DataFrame,
                     start_date: str = "2023-05-17",
                     end_date: str = "2026-03-31") -> pd.DataFrame:
    """
    Ejecuta walk-forward validation entre start_date y end_date.
    Para cada día del periodo:
      1. Predice el precio del día siguiente con los datos hasta ese momento
      2. Hace fine-tuning con el dato real del nuevo día
      3. Registra predicción vs real

    Retorna DataFrame con el log completo.
    """
    print(f"Walk-Forward: {start_date} a {end_date}")
    print("Esto puede tardar varios minutos...")

    close = data["Close"]
    model, scaler = _load_model_and_scaler()

    # Días hábiles en el rango pedido
    bdays = pd.bdate_range(start=start_date, end=end_date)
    bdays = [d for d in bdays if d in close.index or d <= close.index[-1]]

    results = []
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    for i, target_date in enumerate(bdays):
        target_str = target_date.strftime("%Y-%m-%d")

        # Datos conocidos hasta el día anterior a target_date
        known = close[close.index < target_date]
        if len(known) < SEQUENCE_LENGTH + 1:
            continue

        last_60 = known.values[-SEQUENCE_LENGTH:]
        pred = _predict_one_step(model, scaler, last_60)

        # Precio real del target_date (si ya existe)
        if target_date in close.index:
            real = float(close.loc[target_date])
            base = float(known.iloc[-1])
            error_abs = abs(real - pred)
            error_pct = round(error_abs / real * 100, 2)
            direction_correct = int((real > base) == (pred > base))

            results.append({
                "fecha": target_str,
                "prediccion": round(pred, 2),
                "real": round(real, 2),
                "error_abs": round(error_abs, 2),
                "error_pct": error_pct,
                "direction_correct": direction_correct,
            })

            # Fine-tune con los últimos 60 días + el dato real nuevo
            recent = known.values[-(SEQUENCE_LENGTH):].tolist() + [real]
            _finetune(model, scaler, np.array(recent))

        if (i + 1) % 50 == 0:
            evaluated = [r for r in results if "direction_correct" in r]
            if evaluated:
                acc = round(sum(r["direction_correct"] for r in evaluated) / len(evaluated) * 100, 1)
                mae = round(sum(r["error_abs"] for r in evaluated) / len(evaluated), 1)
                print(f"  Progreso: {i+1}/{len(bdays)} | Dir. accuracy: {acc}% | MAE: {mae} pts")

    df = pd.DataFrame(results)
    if not df.empty:
        out_path = DATA_DIR / "walk_forward_results.csv"
        df.to_csv(out_path, index=False)

        direction_acc = round(df["direction_correct"].mean() * 100, 1)
        mae = round(df["error_abs"].mean(), 2)
        rmse = round(np.sqrt((df["error_abs"] ** 2).mean()), 2)

        print(f"\nWalk-Forward completado — {len(df)} dias evaluados")
        print(f"  Direction Accuracy: {direction_acc}%")
        print(f"  MAE:  {mae} pts")
        print(f"  RMSE: {rmse} pts")
        print(f"  Guardado en: {out_path}")

        plot_walk_forward(df)

        # Guardar modelo fine-tuneado
        model.save(MODELS_DIR / "lstm_model_finetuned.keras")
        print("  Modelo fine-tuneado guardado en models/lstm_model_finetuned.keras")

    return df


def plot_walk_forward(df: pd.DataFrame) -> None:
    """Grafica predicciones vs reales del walk-forward."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df["fecha"] = pd.to_datetime(df["fecha"])
    n = min(250, len(df))
    subset = df.tail(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Gráfico de precios
    axes[0].plot(subset["fecha"], subset["real"],
                 color="#1f77b4", lw=1.2, label="Real")
    axes[0].plot(subset["fecha"], subset["prediccion"],
                 color="#ff7f0e", lw=0.9, ls="--", label="Prediccion Walk-Forward")
    axes[0].set_title(f"Walk-Forward: Prediccion vs Real (ultimos {n} dias habiles)")
    axes[0].set_ylabel("Puntos IBEX 35")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Gráfico de errores
    colors = ["#68d391" if c == 1 else "#fc8181"
              for c in subset["direction_correct"]]
    axes[1].bar(subset["fecha"], subset["error_abs"], color=colors, width=1.5, alpha=0.8)
    axes[1].set_title("Error Absoluto por dia (verde = acerto direccion, rojo = fallo)")
    axes[1].set_ylabel("Error (pts)")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    path = PLOTS_DIR / "walk_forward_results.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Grafico guardado: {path}")
