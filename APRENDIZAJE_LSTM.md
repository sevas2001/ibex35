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

## 7. Walk-Forward — El modelo aprende cada día

**Implementado en `src/walk_forward.py`**

En lugar de entrenar una vez con datos hasta 2023 y olvidarse,
el walk-forward reentrena el modelo progresivamente añadiendo cada día real:

```
Iteración 1: train=[2012..2023-05-16] → predice 2023-05-17 → compara con real
Iteración 2: train=[2012..2023-05-17] → predice 2023-05-18 → compara con real
Iteración 3: train=[2012..2023-05-18] → predice 2023-05-19 → compara con real
...
Iteración 727: train=[2012..2026-03-18] → predice 2026-03-19
```

### Fine-tuning incremental

No reentrenamos el modelo desde cero cada día (tardaría horas).
En su lugar hacemos **fine-tuning**: cargamos el modelo existente y
lo entrenamos 5 epochs más con los datos más recientes.

```
modelo_existente.fit(ultimos_60_dias + nuevo_dia_real, epochs=5)
```

Esto es mucho más rápido (~2 segundos por día) y permite que el modelo
se adapte gradualmente a cambios en el mercado.

### Resultados reales del walk-forward (2023-05-17 → 2026-03-19)

| Métrica | Resultado |
|---------|-----------|
| Dias evaluados | 727 |
| Direction Accuracy | **48.4%** |
| MAE | 105.84 pts |
| RMSE | 146.88 pts |

**¿Qué significa 48.4% de direction accuracy?**
El modelo acierta la dirección (sube/baja) en casi 1 de cada 2 días.
Un modelo aleatorio acierta exactamente el 50%. Nuestro modelo está
muy cerca del azar — lo cual es un resultado **honesto y esperado**
para un modelo que solo usa el precio de cierre.

Un modelo que usara múltiples variables (volumen, índices extranjeros,
tipos de interés) podría mejorar este resultado.

---

## 8. Sistema de tracking diario de predicciones

**Implementado en `src/prediction_logger.py`**

Cada vez que alguien consulta la predicción a 5 días en la web:
1. La predicción del **día 1** se guarda automáticamente en `data/prediction_log.csv`
2. Al día siguiente, el sistema descarga el precio real de yfinance
3. Calcula el error y si acertó la dirección
4. El dashboard muestra el historial con ✓/✗

```
prediction_log.csv:
fecha_prediccion | precio_base | prediccion_d1 | real_d1 | error_abs | direction_correct
2026-03-20       | 16905.90    | 16942.58      | ?       | ?         | ?
```

La columna `real_d1` se rellena automáticamente al día siguiente
cuando yfinance publica el precio de cierre.

**direction_correct:**
- `1` → el modelo acertó si el índice subiría o bajaría
- `0` → el modelo falló la dirección
- vacío → aún no hay dato real disponible

---

## 9. Train Fit — ¿El modelo aprendió el pasado?

**Gráfico generado en `data/plots/train_fit.png`**

El train fit muestra las predicciones del modelo sobre los datos de
entrenamiento (2012-2023) comparadas con los valores reales.

**Resultados del train:**
- MAE: 100.79 pts
- RMSE: 138.39 pts

Si el modelo hubiera **memorizado** el train set (overfitting extremo),
el error sería casi cero. El hecho de que el MAE sea ~100 pts indica
que el modelo **generalizó** — aprendió patrones pero no memorizó.

**Comparando train vs test:**

| | MAE | RMSE |
|--|-----|------|
| Train set | 100 pts | 138 pts |
| Test set | 142 pts | 184 pts |

El error sube un ~40% de train a test. Esto es overfitting moderado —
normal y aceptable. Un overfitting severo sería error de 10 pts en train
y 500 pts en test.

---

## 10. Limitaciones honestas del modelo

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

## 11. Para seguir aprendiendo

- **Libro gratuito:** "Dive into Deep Learning" — capítulo de RNN/LSTM
- **Paper original LSTM:** Hochreiter & Schmidhuber (1997) — "Long Short-Term Memory"
- **Mejora del proyecto:** añadir indicadores técnicos (RSI, MACD, Bollinger Bands)
  como features adicionales al LSTM
