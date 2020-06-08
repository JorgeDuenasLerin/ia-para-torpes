import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


tall = tfds.load('oxford_flowers102', split='test+train+validation', as_supervised=True)

num_training_examples = 1020
num_validation_examples = 6149
num_test_examples = 1020
num_classes = 102
num_all = num_training_examples+num_validation_examples+num_test_examples

tall = tall.shuffle(num_all)
#tall = tall.filter(lambda i,l: l < 10)

item_count = 0

for i in tall:
    item_count=item_count+1

ntrain = int(0.8 * item_count)
nval = int(0.2 * item_count)

print('Total: {}'.format(item_count))
print('Train: {}'.format(ntrain))
print('Val: {}'.format(nval))


IMAGE_RES = 300
BATCH_SIZE = 32


@tf.function
def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label


train_batches = tall.take(ntrain).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = tall.skip(ntrain)
validation_batches = validation_batches.take(nval).map(format_image).batch(BATCH_SIZE).prefetch(1)


# MODELO
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


# Modelo
model = Sequential()
# padding -> valid -> sin padding
# 32 filtros
model.add(Conv2D(4, (3, 3), input_shape=(IMAGE_RES, IMAGE_RES, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(160, activation="relu"))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

EPOCHS = 20
#PATIENCE = 5

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

# Stop training when there is no improvement in the validation loss for 5 consecutive epochs
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches#,
 #                   callbacks=[early_stopping]
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