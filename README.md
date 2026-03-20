---
title: IBEX 35 LSTM Predictor
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# IBEX 35 — Predictor LSTM

Dashboard financiero con predicción a 5 días del índice IBEX 35 usando redes LSTM.

- **Backend**: FastAPI + TensorFlow 2.16
- **Modelo**: LSTM (2 capas, 100 neuronas) + ARIMA(5,1,0) como baseline
- **Datos**: yfinance desde 2012 hasta hoy
- **Walk-forward**: 727 días evaluados, MAE 105 pts
