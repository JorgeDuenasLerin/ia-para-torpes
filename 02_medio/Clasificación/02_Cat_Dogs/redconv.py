import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

DIR_PATH="/home/folen/dogs-vs-cats/"
DIR_TRAIN=DIR_PATH+"train"
DIR_VALID=DIR_PATH+"valid"

IMAGE_RES=200
EPOCHS=15
BATCH=256

ntrain=23076
nvalid=1924

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        DIR_TRAIN,
        target_size=(IMAGE_RES, IMAGE_RES),
        batch_size=BATCH,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        DIR_VALID,
        target_size=(IMAGE_RES, IMAGE_RES),
        batch_size=BATCH,
        class_mode='binary')

"""
# Pinta imágenes y comprueba etiqueta
e=next(iter(train_generator))

array_image = e[0][0]
value_label = e[1][0]

plt.imshow(array_image, interpolation='nearest')
plt.title(value_label)
plt.show()
"""


# MODELO
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


# Modelo
model = Sequential()
# padding -> valid -> sin padding
# 32 filtros
model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_RES, IMAGE_RES, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
  optimizer='rmsprop',
  loss='binary_crossentropy',
  metrics=['accuracy'])


history = model.fit(train_generator,
                    steps_per_epoch=ntrain//BATCH,
                    #steps_per_epoch=4000//BATCH,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    validation_data=validation_generator,
                    validation_steps=nvalid//BATCH
                    #validation_steps=800//BATCH
                    )


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()