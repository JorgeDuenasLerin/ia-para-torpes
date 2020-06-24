
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#DIR_PATH="/home/folen/datasets/cuadros/dirs/"
DIR_PATH="/home/folen/datasets/cuadros/concrete/"

DIR_TRAIN=DIR_PATH+"train"
DIR_VALID=DIR_PATH+"valid"

results_subfix="autoconv_conv2_concrete"

IMAGE_RES=128
image_height=IMAGE_RES
image_width=IMAGE_RES

EPOCHS=10
BATCH=32

ntrain = 7736
ntrain = 1296
num_train_images = ntrain

nvalid = 619
nvalid = 108
num_test_images = nvalid


train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        DIR_TRAIN,
        target_size=(IMAGE_RES, IMAGE_RES),
        batch_size=BATCH,
        class_mode='input'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        DIR_VALID,
        target_size=(IMAGE_RES, IMAGE_RES),
        batch_size=BATCH,
        class_mode='input'
)

"""
X = next(iter(train_generator))
plt.imshow(X[0][0], interpolation='nearest')
plt.show()
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose


def make_layer_model(num_hidden_neurons):
    # Define the model
    model = Sequential()
    model.add(Flatten(input_shape=(image_height, image_width, 3)))
    model.add(Dense(num_hidden_neurons*2, activation='relu'))
    model.add(Dense(num_hidden_neurons, activation='relu'))
    model.add(Dense(num_hidden_neurons * 2, activation='relu'))
    model.add(Dense(image_height * image_width * 3, activation='relu'))
    model.add(Reshape((image_height, image_width, 3)))
    model.summary()
    return model


def make_one_layer_model(num_hidden_neurons):
    # Define the model
    model = Sequential()
    model.add(Flatten(input_shape=(image_height, image_width, 3)))
    model.add(Dense(num_hidden_neurons, activation='relu'))
    model.add(Dense(image_height * image_width * 3, activation='relu'))
    model.add(Reshape((image_height, image_width, 3)))
    model.summary()
    return model


def make_conv(num_hidden_neurons):
    model = Sequential()
    f = 64
    model.add(Input(shape=(image_height, image_width, 3)))
    model.add(Conv2D(f, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Conv2D(f//2, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Conv2D(f//4, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(Conv2D(f//8, (3, 3), activation='relu', padding='same'))
    model.add(Flatten(name='flatten')) # 8, 8, 16
    #model.add(Dense(num_hidden_neurons))
    model.add(Dense(num_hidden_neurons))
    #model.add(Dense(num_hidden_neurons))
    model.add(Reshape((16, 16, 4)))

    k=3
    model.add(Conv2DTranspose(f // 8, (k, k), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(f // 4, (k, k), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(f // 2, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    model.add(Conv2DTranspose(f, (k, k), activation='relu'))
    #model.add(Conv2DTranspose(f, (k * 2, k * 2), activation='relu'))
    #model.add(Conv2DTranspose(f, (k * 2, k * 2), activation='relu'))
    #model.add(Conv2DTranspose(f, (k * 2, k * 2), activation='relu'))
    #model.add(Conv2DTranspose(f, (k * 2, k * 2), activation='relu'))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    model.summary()
    return model


def make_conv2(num_hidden_neurons):
    model = Sequential()
    f = 32
    model.add(Input(shape=(image_height, image_width, 3)))
    model.add(Conv2D(f, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Conv2D(f//2, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Conv2D(f//4, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    #model.add(Conv2D(f//8, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(f // 16, (3, 3), activation='relu', padding='same'))
    model.add(Flatten(name='flatten'))
    #model.add(Dense(num_hidden_neurons))
    model.add(Dense(num_hidden_neurons))
    #model.add(Dense(num_hidden_neurons))
    model.add(Reshape((16, 16, 8)))
    k=3
    #model.add(Conv2DTranspose(f // 16, kernel_size=(3, 3), activation='relu', padding='same'))
    #model.add(Conv2DTranspose(f // 8, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(f // 4, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(f // 2, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(f, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    model.summary()
    return model



def plot(X, subfix):
    fig, axs = plt.subplots(1,12, figsize=(17,6))
    for i in range(12):
        axs[i].imshow(X[i], cmap = plt.get_cmap('gray'))
        axs[i].axis('off')
    plt.savefig('pictures'+str(subfix)+'_'+results_subfix+'.png')
    #plt.show()


def biplot(XOri, X, subfix):
    fig, axs = plt.subplots(2, 12, figsize=(128,128))
    for i in range(12):
        axs[0][i].imshow(XOri[i], cmap = plt.get_cmap('gray'))
        axs[0][i].axis('off')
    for i in range(12):
        axs[1][i].imshow(X[i], cmap = plt.get_cmap('gray'))
        axs[1][i].axis('off')
    plt.savefig('pictures'+str(subfix)+'_'+results_subfix+'.png')
    #plt.show()


def one_layer_autoencoder(num_hidden_neurons):
    model = make_conv2(num_hidden_neurons)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        batch_size=BATCH,
        steps_per_epoch=ntrain // BATCH,
        validation_steps=nvalid // BATCH,
        verbose=1
    )

    X = []
    XOri = []
    n = 0
    for img in train_generator:
        XOri.append(img[0][0].reshape(1, image_height, image_width, 3)[0])
        X.append(model.predict(img[0][0].reshape(1, image_height, image_width, 3))[0])
        n = n + 1
        if n == 12:
            break

    biplot(XOri, X, num_hidden_neurons)

    # Guardar el Modelo
    model.save('autoencoder'+str(num_hidden_neurons)+results_subfix+'.h5')


"""
for num_hidden_neurons in [16, 32, 64, 128, 256]:
    one_layer_autoencoder(num_hidden_neurons)
"""


one_layer_autoencoder(2048)#1024)

