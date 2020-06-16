from tensorflow import keras

from skimage import io
import cv2
import matplotlib.pyplot as plt
import math

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform


model_dogs_cats = load_model('conv2_data_augmentation.h5')


IMAGE_RES=300


model_dogs_cats.summary()


#PATH="/home/folen/datasets/dogs-vs-cats/jesus/DogsCats/DogsCats/train/dogs/"
#IMG='dog.592.jpg'
PATH="/home/folen/datasets/dogs-vs-cats/jesus/DogsCats/DogsCats/train/cats/"
IMG='cat.119.jpg'

image = cv2.imread(PATH+IMG, cv2.IMREAD_COLOR)
input_dog = cv2.resize(image, (IMAGE_RES, IMAGE_RES))
plt.imshow(input_dog)
plt.axis('off')
plt.show()

from tensorflow.keras import models

for layer in model_dogs_cats.layers:
    print(layer.name)

import numpy as np

input_dog = input_dog.reshape(1, IMAGE_RES, IMAGE_RES, 3).astype('float32')
input_dog = input_dog / 255


def get_activation(layer):
    output = model_dogs_cats.layers[layer].output  # layer output
    activation_model = models.Model(inputs=model_dogs_cats.input,
                                    outputs=output)
    activation = activation_model.predict(input_dog)
    return (activation)


def plot(activation):
    num_filters = activation.shape[-1]
    size = activation.shape[-2]
    n_col_row = int(math.sqrt(num_filters))
    NUM_COLUMNS = n_col_row # number of images in a row
    # draw filter
    num_rows = num_filters // NUM_COLUMNS
    fig, axs = plt.subplots(num_rows, NUM_COLUMNS, figsize=(size, size))
    for r in range(num_rows):
        for c in range(NUM_COLUMNS):
            filter = r * NUM_COLUMNS + c
            axs[r, c].imshow(activation[0, :, :, filter],
                             cmap=plt.get_cmap('gray'))
            axs[r, c].axis('off')
    plt.tight_layout()
    plt.show()

"""
for layer in (0,2,4,7,10,13):  # max_pooling layers
    activation = get_activation(layer)
    print(model_dogs_cats.layers[layer].name + "  " +
          str(activation.shape))  # (1, height, width, num_filters)
    plot(activation)
"""

activation = get_activation(4)
plot(activation)