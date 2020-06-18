from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from numpy import save

DIR_PATH="/home/folen/datasets/cuadros/concrete/"
DIR_TRAIN=DIR_PATH+"train"
DIR_VALID=DIR_PATH+"valid"

IMAGE_RES=300
EPOCHS=15
BATCH=36 # El data augmentation daba problemas y devolvía un batch menor. Divisor del conjunto de datos

ntrain=648*5
nvalid=54


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_RES, IMAGE_RES,3))
vgg16.trainable = False # no es necesario pero así haríamos que no se entrene en el fit


last_l = vgg16.layers[-1]
last_shape = last_l.output

activation_w = last_shape.shape[1].value
activation_h = last_shape.shape[2].value
last_layer_filter = last_shape.shape[3].value

print ("Activation size: {0}x{1}".format(activation_w, activation_h))
print ("Number of filters: {0}".format(last_layer_filter))


def activations_fromVGG16(source_dir, num_features, augmented=False):
    # Memory for data
    activations = np.zeros((num_features, activation_w, activation_h, last_layer_filter))  # block5_pool shape
    labels = np.zeros((num_features,))

    if augmented == False:
        img_generator = ImageDataGenerator(rescale=1. / 255.)
    else:
        img_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.4,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    batch_size = BATCH

    generator = img_generator.flow_from_directory(
        source_dir,
        target_size=(IMAGE_RES, IMAGE_RES),
        batch_size=batch_size,
        class_mode='binary')

    i = num_features // batch_size - 1
    total_batchs = i
    for input_batch, label_batch in generator:
        # each image enters in the vgg16 model
        output_batch = vgg16.predict(input_batch)
        activations[i * batch_size:(i + 1) * batch_size] = output_batch
        labels[i * batch_size:(i + 1) * batch_size] = label_batch  # labels
        print("Batch: {0}/{1}".format(i, total_batchs))
        i -= 1
        if i == -1:
            break

    return activations, labels


train_data_dir = DIR_TRAIN
validation_data_dir = DIR_VALID

train_activations, train_labels = activations_fromVGG16(train_data_dir, ntrain, True)
validation_activations, validation_labels = activations_fromVGG16(validation_data_dir, nvalid)

save('train_activations_aug.npy', train_activations)
save('train_labels_aug.npy', train_labels)
save('validation_activations_aug.npy', validation_activations)
save('validation_labels_aug.npy', validation_labels)
