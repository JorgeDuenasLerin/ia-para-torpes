import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


tr, ts, va = tfds.load('oxford_flowers102', split=['train', 'test', 'validation'], as_supervised=True)


num_training_examples = 1020
num_validation_examples = 6149
num_test_examples = 1020
num_classes = 102

print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {}'.format(num_validation_examples))
print('Total Number of Test Images: {} \n'.format(num_test_examples))
print('Total Number of Classes: {}'.format(num_classes))


IMAGE_RES = 300

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return tf.reshape(image, [-1]), label


BATCH_SIZE = 32

train_batches = tr.cache().shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = ts.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = va.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

# Modelo
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(400, input_dim=IMAGE_RES*IMAGE_RES*3, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(102, activation='softmax'))

model.summary()

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

EPOCHS = 100

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

