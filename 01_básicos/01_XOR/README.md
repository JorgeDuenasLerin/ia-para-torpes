# Puerta XOR con IA

Pasos para crear una red de neuronas que se comporte como una puerta XOR

Objetivos:
1. Hacer funcionar el entorno
2. Afianzar conceptos

Basado en este [tutorial](https://www.aprendemachinelearning.com/una-sencilla-red-neuronal-en-python-con-keras-y-tensorflow/)

## Tabla de verdad

| A | B | A XOR B |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

## Conjunto de datos
En este sencillo ejemplo tenemos todas las pobilidades dentro del conjunto de datos. No tiene sentido hacer una partcición en: entrenamiento, validación, test.
Todos los datos se pueden generar ¿Nomeclatura datos sintéticos?

```python
# Tabla de valores entrada y valores esperados
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")
```


## Modelo

Secuencial: una detrás de otra

Capas:
1. Entrada: 2 neuronas
2. Oculta: 8 neuronas
3. Salida: 1 nuerona (0 o 1)

```python
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

Épocas: 1000

## Código resultado

```python

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Tabla de valores entrada y valores esperados
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000)

# evaluamos el modelo
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())

```
