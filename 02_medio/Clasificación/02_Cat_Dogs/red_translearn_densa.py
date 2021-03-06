
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from numpy import load
from numpy import save

SUBFIX='_sliced'

train_activations = load('train_activations'+SUBFIX+'.npy')
train_labels = load('train_labels'+SUBFIX+'.npy')
validation_activations = load('validation_activations'+SUBFIX+'.npy')
validation_labels = load('validation_labels'+SUBFIX+'.npy')


IMAGE_RES = 224
EPOCHS = 50
BATCH = 64

#activation_w = 7
#activation_h = 7
#last_layer_filter = 512


activation_w = 14
activation_h = 14
last_layer_filter = 256


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(512, (3, 3), input_shape=(activation_w, activation_h, last_layer_filter), padding="valid", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="valid", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="valid", activation="relu"))
model.add(Conv2D(512, (3, 3), padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), padding="valid", activation="relu"))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

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
model.save('conv2_transfer_learning.h5')
# Guardar evolución entrenamiento
save('conv2_transfer_learning_acc.npy', acc)
save('conv2_transfer_learning_val_acc.npy', val_acc)
save('conv2_transfer_learning_loss.npy', loss)
save('conv2_transfer_learning_val_loss.npy', val_loss)

