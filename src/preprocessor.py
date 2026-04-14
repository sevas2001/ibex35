import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL

DATA_DIR   = Path(__file__).parent.parent / "data"
PLOTS_DIR  = DATA_DIR / "plots"
MODELS_DIR = Path(__file__).parent.parent / "models"

SEQUENCE_LENGTH = 60
TRAIN_RATIO     = 0.80
N_FEATURES      = 1   # Close (univariado — modelos LSTM/GRU originales)
N_FEATURES_V2   = 7   # 7 indicadores tecnicos (v2, resultado: peor que v1 por multicolinealidad)
N_FEATURES_V3   = 5   # close, log_return, ma50_ratio, crisis_regime, crisis_decay
N_FEATURES_V4   = 8   # SOTA: log_return (TARGET), close_norm, sma20_ratio, sma50_ratio,
                       #       rsi14, macd_norm, bb_width, atr14_norm

# Eventos historicos del IBEX 35 para features de crisis (v3).
# Fechas basadas en declaraciones publicas — sin data leakage.
# Cada entrada: (fecha_inicio, fecha_fin, descripcion)
IBEX_CRISIS_EVENTS = [
    ("2008-09-15", "2009-03-09",  "gfc"),             # Lehman → mínimo mercado bear
    ("2010-04-01", "2012-07-26",  "sovereign_debt"),   # Crisis deuda soberana española
    ("2012-05-09", "2012-12-31",  "bankia"),           # Rescate Bankia
    ("2016-06-23", "2016-07-15",  "brexit"),           # Referéndum Brexit
    ("2017-10-01", "2017-12-21",  "catalonia"),        # Referéndum independencia Cataluña
    ("2020-02-21", "2020-03-23",  "covid_crash"),      # COVID crash (-38% en 30 días)
    ("2020-11-09", "2021-06-30",  "covid_recovery"),   # Rebote vacunas Pfizer
    ("2022-02-24", "2022-09-30",  "ukraine"),          # Invasión Ucrania
]


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DataFrame con solo Close (modelo univariado LSTM/GRU original).
    """
    df = pd.DataFrame(index=data.index)
    df["Close"] = data["Close"]
    return df.dropna()


def compute_features_v2(data: pd.DataFrame) -> pd.DataFrame:
    """
    7 features derivadas exclusivamente del precio de cierre (sin Volume).
    Columna 0 siempre es 'close' (necesario para inverse_transform de predicciones).

      0 close:       precio de cierre (target inverse_transform)
      1 log_return:  ln(Close_t / Close_{t-1}) — estacionariedad
      2 ma50_ratio:  Close / MA(50) — posicion relativa a tendencia larga
      3 rsi14:       RSI(14) / 100 — momentum normalizado [0,1]
      4 macd_norm:   (EMA12 - EMA26) / Close — cruce de medias normalizado
      5 bb_width:    (2 * std20) / Close — anchura de Bollinger normalizada
      6 ema20_ratio: Close / EMA(20) — posicion relativa a tendencia corta
    """
    df = pd.DataFrame(index=data.index)
    close = data["Close"]

    # 0 — close
    df["close"] = close

    # 1 — log_return
    df["log_return"] = np.log(close / close.shift(1))

    # 2 — ma50_ratio
    ma50 = close.rolling(50).mean()
    df["ma50_ratio"] = close / ma50

    # 3 — RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi14"] = (100 - 100 / (1 + rs)) / 100.0

    # 4 — MACD normalizado
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_norm"] = (ema12 - ema26) / close

    # 5 — Bollinger Band width normalizada
    std20 = close.rolling(20).std()
    df["bb_width"] = (2 * std20) / close

    # 6 — EMA20 ratio
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["ema20_ratio"] = close / ema20

    return df.dropna()


def split_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal 80/20 sin shuffle para respetar el orden cronologico."""
    split_idx = int(len(features) * TRAIN_RATIO)
    train = features.iloc[:split_idx]
    test  = features.iloc[split_idx:]
    print(f"Train: {len(train)} dias ({train.index[0].date()} — {train.index[-1].date()})")
    print(f"Test:  {len(test)} dias ({test.index[0].date()} — {test.index[-1].date()})")
    return train, test


def fit_scaler(train_features: pd.DataFrame) -> MinMaxScaler:
    """
    Ajusta MinMaxScaler sobre los 4 features de entrenamiento.
    CRITICO: nunca hacer fit sobre todo el dataset (data leakage).
    La columna 0 (Close) se usa para inverse_transform de predicciones.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features.values)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("Scaler (4 features) guardado en models/scaler.pkl")
    return scaler


def create_sequences(features_scaled: np.ndarray,
                     seq_length: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte el array escalado en pares (X, y) para LSTM multivariate.
      X[i]: ventana (seq_length, N_FEATURES)
      y[i]: Close escalado del dia siguiente (columna 0)
    """
    X, y = [], []
    for i in range(seq_length, len(features_scaled)):
        X.append(features_scaled[i - seq_length:i])   # (60, 4)
        y.append(features_scaled[i, 0])                # target = Close
    return np.array(X), np.array(y)


def fit_scaler_v2(train_features: pd.DataFrame) -> MinMaxScaler:
    """
    Ajusta MinMaxScaler sobre los 7 features de entrenamiento (v2).
    CRITICO: nunca hacer fit sobre todo el dataset (data leakage).
    Columna 0 (close) se usa para inverse_transform de predicciones.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features.values)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler_v2.pkl")
    print(f"Scaler v2 ({train_features.shape[1]} features) guardado en models/scaler_v2.pkl")
    return scaler


def preprocess_v2(data: pd.DataFrame) -> dict:
    """
    Pipeline completo de preprocesamiento con 7 features (v2).
    Entrena scaler_v2.pkl y retorna dict con X_train, y_train, X_test, y_test.
    """
    features = compute_features_v2(data)
    train, test = split_data(features)
    scaler = fit_scaler_v2(train)

    train_scaled = scaler.transform(train.values)
    full_scaled  = scaler.transform(features.values)

    X_train, y_train = create_sequences(train_scaled)

    # Test necesita los ultimos SEQUENCE_LENGTH dias de train como contexto
    test_input = full_scaled[len(train) - SEQUENCE_LENGTH:]
    X_test, y_test = create_sequences(test_input)

    output = {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_test":     X_test,
        "y_test":     y_test,
        "train_size": len(train),
        "test_size":  len(test),
        "scaler":     scaler,
        "features":   features,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "preprocessed_data_v2.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test,  y_test=y_test,
    )
    print(f"\nDatos v2 guardados en data/preprocessed_data_v2.npz")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"Features: {list(features.columns)}")
    return output


def _build_crisis_features(index: pd.DatetimeIndex,
                            decay_lambda: float = 0.05) -> pd.DataFrame:
    """
    Genera dos features de eventos historicos para el IBEX 35:

      crisis_regime : dummy binario  — 1 si la fecha cae dentro de una crisis
      crisis_decay  : decaimiento exponencial desde el inicio del evento activo
                      f(t) = exp(-lambda * dias_desde_inicio)
                      Si hay multiples eventos solapados, se toma el maximo.

    Sin data leakage: fechas son hechos publicos historicos, no derivadas de precios.
    """
    regime = pd.Series(0.0, index=index, dtype=float)
    decay  = pd.Series(0.0, index=index, dtype=float)

    for start_str, end_str, _ in IBEX_CRISIS_EVENTS:
        t0   = pd.Timestamp(start_str)
        t1   = pd.Timestamp(end_str)
        mask = (index >= t0) & (index <= t1)

        regime[mask] = 1.0

        days_since = (index[mask] - t0).days.astype(float)
        decay_vals = np.exp(-decay_lambda * days_since)
        decay[mask] = np.maximum(decay[mask], decay_vals)

    return pd.DataFrame({"crisis_regime": regime, "crisis_decay": decay}, index=index)


def compute_features_v3(data: pd.DataFrame,
                        decay_lambda: float = 0.05) -> pd.DataFrame:
    """
    5 features para v3: lo mejor de v1 + eventos historicos, sin redundancias.

    Columnas (orden importante para inverse_transform):
      0 close          — precio de cierre (target)
      1 log_return     — ln(Close_t/Close_{t-1}), estacionariedad
      2 ma50_ratio     — Close / MA(50), tendencia de largo plazo
      3 crisis_regime  — dummy: 1 si fecha en ventana de crisis historica
      4 crisis_decay   — intensidad exponencial del evento activo

    Eliminados de v2 por multicolinealidad: rsi14, macd_norm, bb_width, ema20_ratio.
    """
    df    = pd.DataFrame(index=data.index)
    close = data["Close"]

    df["close"]      = close
    df["log_return"] = np.log(close / close.shift(1))

    ma50 = close.rolling(50).mean()
    df["ma50_ratio"] = close / ma50

    crisis = _build_crisis_features(data.index, decay_lambda=decay_lambda)
    df["crisis_regime"] = crisis["crisis_regime"]
    df["crisis_decay"]  = crisis["crisis_decay"]

    return df.dropna()


def fit_scaler_v3(train_features: pd.DataFrame) -> MinMaxScaler:
    """
    Ajusta MinMaxScaler sobre los 5 features v3.
    CRITICO: fit solo sobre training set (sin data leakage).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features.values)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler_v3.pkl")
    print(f"Scaler v3 ({train_features.shape[1]} features) guardado en models/scaler_v3.pkl")
    return scaler


def preprocess_v3(data: pd.DataFrame) -> dict:
    """
    Pipeline completo v3: 5 features seleccionadas + eventos historicos.
    Guarda scaler_v3.pkl y preprocessed_data_v3.npz.
    """
    features = compute_features_v3(data)
    train, test = split_data(features)
    scaler = fit_scaler_v3(train)

    train_scaled = scaler.transform(train.values)
    full_scaled  = scaler.transform(features.values)

    X_train, y_train = create_sequences(train_scaled)

    test_input = full_scaled[len(train) - SEQUENCE_LENGTH:]
    X_test, y_test = create_sequences(test_input)

    output = {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_test":     X_test,
        "y_test":     y_test,
        "train_size": len(train),
        "test_size":  len(test),
        "scaler":     scaler,
        "features":   features,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR / "preprocessed_data_v3.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test)
    print(f"\nDatos v3 guardados en data/preprocessed_data_v3.npz")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"Features v3: {list(features.columns)}")
    return output


def compute_features_v4(data: pd.DataFrame) -> pd.DataFrame:
    """
    8 features para v4 — log_return como TARGET (serie estacionaria).

    Columnas (orden critico para el scaler):
      0 log_return   — TARGET: ln(Close_t/Close_{t-1}), estacionario
      1 close_norm   — Close / max_rolling(252, min=60) — nivel sin tendencia
      2 sma20_ratio  — Close / SMA(20)
      3 sma50_ratio  — Close / SMA(50)
      4 rsi14        — RSI(14)/100, momentum [0,1]
      5 macd_norm    — (EMA12-EMA26)/Close
      6 bb_width     — 2*std20/Close, volatilidad relativa
      7 atr14_norm   — ATR(14)/Close, volatilidad real (requiere High, Low)

    Diferencia clave vs v3: col 0 es log_return (no close).
    El modelo aprende retornos logaritmicos (estacionarios).
    Los precios reales se reconstruyen con reconstruct_prices_from_returns().
    """
    df    = pd.DataFrame(index=data.index)
    close = data["Close"]

    # 0 — log_return (TARGET, estacionario)
    df["log_return"] = np.log(close / close.shift(1))

    # 1 — close_norm: nivel de precio normalizado respecto a maximo 1 año
    df["close_norm"] = close / close.rolling(252, min_periods=60).max()

    # 2 — sma20_ratio
    df["sma20_ratio"] = close / close.rolling(20).mean()

    # 3 — sma50_ratio
    df["sma50_ratio"] = close / close.rolling(50).mean()

    # 4 — RSI(14) normalizado
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi14"] = (100 - 100 / (1 + rs)) / 100.0

    # 5 — MACD normalizado
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_norm"] = (ema12 - ema26) / close

    # 6 — Bollinger Band width
    std20 = close.rolling(20).std()
    df["bb_width"] = (2 * std20) / close

    # 7 — ATR(14) normalizado (requiere columnas High y Low)
    high       = data["High"]
    low        = data["Low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr14_norm"] = atr14 / close

    return df.dropna()


def fit_scaler_v4(train_features: pd.DataFrame) -> MinMaxScaler:
    """
    Ajusta MinMaxScaler sobre los 8 features v4.
    CRITICO: fit solo sobre training set (sin data leakage).
    Col 0 es log_return — el scaler captura el rango historico de retornos diarios.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features.values)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler_v4.pkl")
    print(f"Scaler v4 ({train_features.shape[1]} features) guardado en models/scaler_v4.pkl")
    return scaler


def preprocess_v4(data: pd.DataFrame) -> dict:
    """
    Pipeline de preprocesamiento v4: log_return como TARGET.

    NO genera X_train/y_train directamente — la creacion de secuencias se hace
    externamente con create_sequences_direct_v4() porque necesita anchor_prices
    (precios reales para reconstruir predicciones via exp()).

    Retorna:
      full_scaled:  (N, 8) array escalado completo
      train_scaled: (split_idx, 8)
      split_idx:    int — indice de corte train/test
      scaler:       MinMaxScaler ajustado sobre train
      features:     DataFrame original (para debug / prediccion futura)
      close_real:   array de Close sin escalar alineado con features.index
    """
    features = compute_features_v4(data)
    train, test = split_data(features)
    scaler = fit_scaler_v4(train)

    full_scaled  = scaler.transform(features.values)
    train_scaled = full_scaled[:len(train)]

    # close_real: precios reales alineados con el indice de features (post-dropna)
    close_real = data["Close"].reindex(features.index).values

    lr = features["log_return"]
    print(f"\nFeatures v4 ({features.shape[1]} cols): {list(features.columns)}")
    print(f"Train: {len(train)} dias | Test: {len(test)} dias")
    print(f"log_return — media: {lr.mean():.6f}  std: {lr.std():.5f}  "
          f"min: {lr.min():.4f}  max: {lr.max():.4f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "preprocessed_data_v4.npz",
        full_scaled=full_scaled,
        train_scaled=train_scaled,
        close_real=close_real,
    )
    print("Datos v4 guardados en data/preprocessed_data_v4.npz")

    return {
        "full_scaled":  full_scaled,
        "train_scaled": train_scaled,
        "split_idx":    len(train),
        "scaler":       scaler,
        "features":     features,
        "close_real":   close_real,
    }


def decompose_series(data: pd.DataFrame) -> None:
    """Descomposicion STL: tendencia, estacionalidad y residuo."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    close = data["Close"].asfreq("B").ffill()

    stl    = STL(close, period=252, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(close.index, result.observed, color="#1f77b4", lw=0.8)
    axes[0].set_title("Serie original")
    axes[1].plot(close.index, result.trend, color="#ff7f0e", lw=0.8)
    axes[1].set_title("Tendencia")
    axes[2].plot(close.index, result.seasonal, color="#2ca02c", lw=0.8)
    axes[2].set_title("Estacionalidad (anual)")
    axes[3].plot(close.index, result.resid, color="#d62728", lw=0.5, alpha=0.7)
    axes[3].set_title("Residuo")

    for ax in axes:
        ax.grid(alpha=0.3)
    plt.suptitle("Descomposicion STL — IBEX 35", fontsize=13, y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / "stl_decomposition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Descomposicion STL guardada: {path}")


def preprocess(data: pd.DataFrame) -> dict:
    """
    Pipeline completo de preprocesamiento multivariate (4 features).
    Retorna dict con X_train, y_train, X_test, y_test y scaler.
    """
    features = compute_features(data)
    train, test = split_data(features)
    scaler = fit_scaler(train)

    train_scaled = scaler.transform(train.values)
    full_scaled  = scaler.transform(features.values)

    X_train, y_train = create_sequences(train_scaled)

    # Test necesita los ultimos SEQUENCE_LENGTH dias de train como contexto
    test_input = full_scaled[len(train) - SEQUENCE_LENGTH:]
    X_test, y_test = create_sequences(test_input)

    output = {
        "X_train":    X_train,
        "y_train":    y_train,
        "X_test":     X_test,
        "y_test":     y_test,
        "train_size": len(train),
        "test_size":  len(test),
        "scaler":     scaler,
        "features":   features,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_DIR / "preprocessed_data.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test,  y_test=y_test,
    )
    print(f"\nDatos preprocesados guardados en data/preprocessed_data.npz")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"Features: {list(features.columns)}")
    return output
