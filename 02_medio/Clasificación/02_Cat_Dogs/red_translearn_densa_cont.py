
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from numpy import load

SUBFIX='_sliced'

train_activations = load('train_activations'+SUBFIX+'.npy')
train_labels = load('train_labels'+SUBFIX+'.npy')
validation_activations = load('validation_activations'+SUBFIX+'.npy')
validation_labels = load('validation_labels'+SUBFIX+'.npy')


IMAGE_RES = 224
EPOCHS = 15
BATCH = 64

#activation_w = 7
#activation_h = 7
#last_layer_filter = 512


activation_w = 14
activation_h = 14
last_layer_filter = 256


from tensorflow.keras.models import load_model
model = load_model('conv2_transfer_learning.h5')

model.summary()

history = model.fit(
                    train_activations,
                    train_labels,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    validation_data=(validation_activations, validation_labels),
                    verbose=1
                )


print(history)

acc = history.history['acc']
val_acc = history.history['val_acc']

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

# Guardar el Modelo
model.save('conv2_transfer_learning_cont.h5')
