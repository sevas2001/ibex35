"""
IBEX 35 — Pipeline Fase 1: Carga y Preprocesamiento de Datos
"""
from src.data_loader import load_ibex35
from src.visualizer import run_eda, plot_close_price, plot_volume
from src.preprocessor import preprocess, decompose_series


def main():
    print("=" * 50)
    print("  IBEX 35 — Predicción LSTM")
    print("  Fase 1: Datos y Preprocesamiento")
    print("=" * 50)

    # 1. Cargar datos
    data = load_ibex35()

    # 2. EDA
    run_eda(data)

    # 3. Visualizaciones
    plot_close_price(data)
    plot_volume(data)

    # 4. Descomposición STL
    decompose_series(data)

    # 5. Preprocesamiento para LSTM
    result = preprocess(data)

    print("\nFase 1 completada.")
    print(f"  Train samples: {result['X_train'].shape[0]}")
    print(f"  Test samples:  {result['X_test'].shape[0]}")
    print("\nSiguiente paso: ejecutar Fase 2 — Modelos ARIMA + LSTM")


if __name__ == "__main__":
    main()
