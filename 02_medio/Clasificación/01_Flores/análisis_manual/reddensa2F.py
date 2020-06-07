import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

tr, ts = tfds.load('oxford_flowers102', split=['test', 'train+validation'], as_supervised=True)

num_training_examples = 1020
num_validation_examples = 6149
num_test_examples = 1020
num_classes=10


IMAGE_RES = 500

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return tf.reshape(image, [-1]), label


BATCH_SIZE = 32

# CAmbiado y filtrado
train_batches = tr.filter(lambda i,l: l < 10).cache().shuffle(num_test_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = ts.filter(lambda i,l: l < 10).cache().map(format_image).batch(BATCH_SIZE).prefetch(1)


# Modelo
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200, input_dim=IMAGE_RES*IMAGE_RES*3, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.summary()

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  #loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

EPOCHS = 15

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

"""
# Stop training when there is no improvement in the validation loss for 5 consecutive epochs
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    callbacks=[early_stopping])
"""




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

