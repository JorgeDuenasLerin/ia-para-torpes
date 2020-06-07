import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt


def draw_array(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()


def draw_tensor(tsimg, label):
    draw_array(np.array(tsimg[label]))


def getfirst(ds):
    e = next(iter(ds))
    draw_tensor(e, 'image')


tr, ts, va = tfds.load('oxford_flowers102', split=['train', 'test', 'validation'], as_supervised=True)


def resize_tensor_image(t, label, size):
    new = tf.image.resize(t[label], size)
    new = tf.cast(new, dtype=np.uint8)
    new = tf.reshape(new, [-1])
    t[label] = new
    return t


size = 50


tr = tr.map(lambda e: resize_tensor_image(e, 'image', [size, size]))
ts = ts.map(lambda e: resize_tensor_image(e, 'image', [size, size]))
va = va.map(lambda e: resize_tensor_image(e, 'image', [size, size]))




def extract_img_val_to_array(e):
    tensor_img = e['image']
    return tensor_img


def extract_label_val(e):
    return e['label']


X_train = tr.map(lambda e: extract_img_val_to_array(e))
y_train = tr.map(lambda e: extract_label_val(e))

X_test = ts.map(lambda e: extract_img_val_to_array(e))
y_test = ts.map(lambda e: extract_label_val(e))

"""
X_train = list(X_train)
y_train = list(y_train)
X_test = list(X_test)
y_test = list(y_test)



print(len(X_train))
print(len(y_train))



print(type(X_train))
print(X_train)
for X in X_train:
    print (X)

print(type(y_train))
print(y_train)
for y in y_train:
    print (y)



"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(500, input_dim=50*50*3, activation='relu'))
model.add(Dense(102, activation='softmax'))
print(model.summary())


EPOCHS = 50
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=EPOCHS, verbose=0)
#history = model.fit(X_train, epochs=EPOCHS, verbose=0)
#history = model.fit(tr, validation_data=ts, epochs=EPOCHS, batch_size=256, verbose=0)

def plot(h):
    LOSS = 0; ACCURACY = 1
    training = np.zeros((2,EPOCHS)); testing = np.zeros((2,EPOCHS))
    training[LOSS] = h.history['loss']
    testing[LOSS] = h.history['val_loss']
    # validation loss
    training[ACCURACY] = h.history['acc']
    testing[ACCURACY] = h.history['val_acc']
    # validation accuracy
    epochs = range(1,EPOCHS+1)
    fig, axs = plt.subplots(1,2, figsize=(17,5))
    for i, label in zip((LOSS, ACCURACY),('loss', 'accuracy')):
        axs[i].plot(epochs, training[i], 'b-', label='Training ' + label)
        axs[i].plot(epochs, testing[i], 'y-', label='Test ' + label)
        axs[i].set_title('Training and test ' + label)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(label)
        axs[i].legend()
    plt.show()

plot(history)