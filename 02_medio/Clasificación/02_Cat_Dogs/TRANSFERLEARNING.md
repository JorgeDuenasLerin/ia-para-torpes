# Transferencia de aprendizaje

Existen modelo de rede ya precargadas que podemos incluir en nuestros modelos.

Hay una red famosa VGG16 que ha clasificado 1000 clases con miles de imágenes. Podemos aprovechar todos los filtros que ha aprendido y ponerlo en nuestro modelo. Al final pondremos nuestra capa densa para la clasificación de perros/gatos

Seguimos con el data augmentation que realizamos en la parte para enriquecer el conjunto de entrenamiento.

```python
from tensorflow.keras.applications import VGG16
vgg16 = VGG16(
            weights='imagenet', # filtros de imagenet
            include_top=False, # sin capa densa
            input_shape=(IMAGE_RES, IMAGE_RES,3)
        )
vgg16.trainable = False # 

# Modelo
model = Sequential()
# padding -> valid -> sin padding
# 32 filtros

model.add(vgg16)

model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
``` 


Modelo:
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 7, 7, 512)         14714688  
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
dense (Dense)                (None, 400)               10035600  
_________________________________________________________________
dense_1 (Dense)              (None, 100)               40100     
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 101       
=================================================================
Total params: 24,790,489
Trainable params: 10,075,801
Non-trainable params: 14,714,688
_________________________________________________________________
```

Podemos ver como los parámetros que vamos a entrenar son los de nuestras capas densas.

Tiempo de entrenamiento: 3 horas

Resultados:
```python
Se está ejecutando...
```
