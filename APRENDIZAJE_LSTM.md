# Guía de Aprendizaje: Predicción del IBEX 35 con LSTM

---

## 1. ¿Por qué LSTM y no una red neuronal normal?

Una red neuronal estándar (Dense) trata cada entrada de forma independiente.
Si le das el precio de hoy, no recuerda nada de ayer.

Las **Redes Recurrentes (RNN)** fueron diseñadas para datos secuenciales:
cada neurona recibe el input actual Y el estado oculto del paso anterior.

```
RNN básica:
  precio_lunes ──> [neurona] ──> estado_lunes
                                     │
  precio_martes ──> [neurona] ──> estado_martes
                                     │
  precio_miércoles ──> [neurona] ──> predicción
```

**El problema de las RNN básicas:** el "gradiente que desaparece".
Al entrenar con muchos pasos de tiempo, el aprendizaje de eventos lejanos
se vuelve prácticamente cero — la red olvida lo que pasó hace 30 días.

**LSTM resuelve esto** con 3 "puertas" que controlan qué recordar y qué olvidar:

| Puerta | Función |
|--------|---------|
| **Forget gate** | Decide qué información del pasado descartar |
| **Input gate** | Decide qué información nueva guardar |
| **Output gate** | Decide qué información pasar al siguiente paso |

Esto permite capturar patrones de hace semanas o meses, lo que es clave
en series financieras donde las tendencias duran tiempo.

---

## 2. Preprocesamiento — Por qué cada paso

### 2.1 Normalización MinMaxScaler → rango [0, 1]

Las redes neuronales aprenden mediante gradientes. Si los valores son muy
grandes (el IBEX está en ~10.000 pts), los gradientes se vuelven inestables.

**CRÍTICO — Data leakage:**
El scaler se ajusta SOLO sobre los datos de entrenamiento (train).
Si lo ajustamos sobre todo el dataset, estamos dándole información del futuro
al modelo durante el entrenamiento → métricas artificialmente buenas que
no se replican en producción.

```python
# MAL (data leakage)
scaler.fit(todos_los_datos)

# BIEN
scaler.fit(solo_train)
scaler.transform(train)  # con parámetros del train
scaler.transform(test)   # con los MISMOS parámetros del train
```

### 2.2 Split temporal 80/20

A diferencia de otros problemas de ML, aquí NO se puede hacer shuffle.
El tiempo tiene un orden causal: el futuro no puede influir en el pasado.

```
|──────── TRAIN (80%) ────────|──── TEST (20%) ────|
2012                        2023                 2026
```

### 2.3 Ventanas de 60 días (sequences)

El LSTM no recibe un precio, recibe una "ventana" de 60 días consecutivos:

```
Input:  [precio_día_1, precio_día_2, ..., precio_día_60]  → shape: (60, 1)
Output: precio_día_61
```

Elegimos 60 días (~3 meses bursátiles) porque:
- Captura tendencias de medio plazo
- Suficiente contexto sin hacer el modelo demasiado complejo

### 2.4 Descomposición STL

Separa la serie en 3 componentes para entenderla mejor:
- **Tendencia**: dirección general a largo plazo
- **Estacionalidad**: patrones que se repiten anualmente
- **Residuo**: lo que no explican ni tendencia ni estacionalidad (ruido + shocks)

En el IBEX el residuo es grande → los mercados tienen mucho ruido aleatorio.

---

## 3. Arquitectura del modelo

```
Input: (60 días, 1 feature)
         │
    LSTM(100 neuronas, return_sequences=True)
         │  ← devuelve secuencia completa para la siguiente capa LSTM
    Dropout(0.2)
         │  ← desactiva 20% de neuronas aleatoriamente → evita overfitting
    LSTM(100 neuronas, return_sequences=False)
         │  ← solo devuelve el último estado
    Dropout(0.2)
         │
    Dense(50, relu)
         │  ← capa intermedia para aprender combinaciones no lineales
    Dense(1)
         │
    Predicción: precio del día 61
```

**¿Por qué 2 capas LSTM?**
La primera aprende patrones de bajo nivel (fluctuaciones cortas).
La segunda aprende patrones de alto nivel (tendencias más amplias).

**¿Por qué Dropout?**
Regularización: durante el entrenamiento "apaga" neuronas al azar.
Esto fuerza a la red a no depender de neuronas específicas → generaliza mejor.

**¿Por qué EarlyStopping?**
Monitoriza la pérdida en validación. Si lleva 10 épocas sin mejorar, para.
Evita que el modelo memorice el train set en lugar de aprender.

---

## 4. Entrenamiento

```
Función de pérdida: MSE (Error Cuadrático Medio)
Optimizador: Adam (adaptativo, ajusta el learning rate automáticamente)
Epochs: hasta 100 (con EarlyStopping)
Batch size: 32 (procesa 32 ventanas a la vez)
```

**¿Qué es una época?**
Una pasada completa por todo el dataset de entrenamiento.
En cada época el modelo actualiza sus pesos para reducir el error.

**¿Qué es el batch size?**
En lugar de actualizar los pesos tras cada ejemplo (muy lento) o tras ver
todos los datos (puede quedarse atascado), actualizamos cada 32 ejemplos.

---

## 5. Predicción a 5 días — Estrategia Autoregresiva

Para predecir 5 días no tenemos los precios reales futuros.
Usamos una estrategia **autoregresiva**: cada predicción se convierte en input:

```
Día 1: input=[d-60...d-1]        → pred_1
Día 2: input=[d-59...d-1, pred_1] → pred_2
Día 3: input=[d-58...d-1, pred_1, pred_2] → pred_3
...
```

**Limitación importante:** el error se acumula.
La predicción del día 1 es más fiable que la del día 5.
Por eso el modelo tiende a "regresar a la media" — las predicciones
lejanas convergen hacia el nivel promedio reciente.

---

## 6. Métricas — ¿Cómo sé si el modelo es bueno?

### Las métricas del proyecto

| Métrica | ARIMA | LSTM | Significado |
|---------|-------|------|-------------|
| **RMSE** | 121 pts | 184 pts | Error típico en puntos del índice |
| **MAE**  | 86 pts  | 142 pts | Error absoluto medio |
| **MSE**  | 14.736  | 34.099 | Penaliza errores grandes |

Con el IBEX en ~10.000-17.000 puntos, un RMSE de 121-184 pts representa
un error del **0.7-1.1%** — razonable para predicción a 1 día.

### ¿Por qué ARIMA gana al LSTM aquí?

Esto es normal y no significa que LSTM sea malo. El ARIMA es mejor en:
- Series con una sola variable (solo el precio)
- Predicciones a muy corto plazo (1-5 días)
- Series con patrones lineales bien definidos

El LSTM brillaría con:
- Múltiples variables (volumen, otros índices, noticias, etc.)
- Patrones no lineales complejos
- Horizontes más largos

### ¿Cómo evaluar si es "útil" en la práctica?

Más allá del RMSE, en finanzas se usa la **dirección**:
¿El modelo predice correctamente si el índice sube o baja?

```
Accuracy direccional = aciertos de dirección / total de días
```

Un modelo que acierta >55% la dirección ya es útil.
Un modelo aleatorio acierta ~50%.

---

## 7. ¿El modelo aprende con nuevos datos?

**Actualmente: NO.** El modelo está entrenado y guardado estáticamente.
Cada vez que alguien pide una predicción:
1. Se descargan los últimos 60 días de yfinance (datos frescos)
2. Se aplican al modelo ya entrenado
3. Se genera la predicción

**Limitación:** el modelo nunca actualiza sus pesos con datos nuevos.
Si el mercado cambia de régimen (crisis, cambio de política monetaria),
el modelo puede quedarse desactualizado.

### Para mejorar esto (trabajo futuro)

**Opción 1 — Reentrenamiento periódico**
Programar un job semanal que descargue los datos nuevos y reentrene el modelo.

**Opción 2 — Online Learning**
Actualizar los pesos del modelo con cada nuevo dato del día.
Más complejo pero el modelo se adapta continuamente.

**Opción 3 — Walk-forward validation**
Técnica más honesta de evaluación: entrena hasta el día X, predice X+1,
añade X+1 al train, predice X+2, etc. Simula el uso real.

---

## 8. Limitaciones honestas del modelo

1. **Los mercados no son predecibles** con alta precisión.
   El IBEX refleja millones de decisiones humanas + eventos imprevisibles.

2. **Solo usamos el precio de cierre.** Un modelo más robusto incluiría:
   - Volumen de negociación
   - Otros índices (S&P500, DAX)
   - Tipos de interés del BCE
   - Índice de volatilidad (VIX)
   - Sentimiento de noticias (NLP)

3. **Error acumulativo en predicciones largas.**
   La predicción del día 5 es mucho menos fiable que la del día 1.

4. **No constituye asesoramiento financiero.**
   Este modelo es una herramienta académica de aprendizaje.

---

## 9. Para seguir aprendiendo

- **Libro gratuito:** "Dive into Deep Learning" — capítulo de RNN/LSTM
- **Paper original LSTM:** Hochreiter & Schmidhuber (1997) — "Long Short-Term Memory"
- **Mejora del proyecto:** añadir indicadores técnicos (RSI, MACD, Bollinger Bands)
  como features adicionales al LSTM
