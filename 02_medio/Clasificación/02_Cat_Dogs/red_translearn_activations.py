
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from numpy import save

DIR_PATH="/home/folen/datasets/dogs-vs-cats/"
DIR_TRAIN=DIR_PATH+"train"
DIR_VALID=DIR_PATH+"valid"

IMAGE_RES=224
EPOCHS=15
BATCH=64

ntrain=23076
nvalid=1924

#ntrain=2300
#nvalid=190


from tensorflow.keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_RES, IMAGE_RES,3))
vgg16.trainable = False # no es necesario pero así haríamos que no se entrene en el fit

last_l = vgg16.layers[-1]
last_shape = last_l.output

activation_w = last_shape.shape[1].value
activation_h = last_shape.shape[2].value
last_layer_filter = last_shape.shape[3].value


print ("Activation size: {0}x{1}".format(activation_w, activation_h))
print ("Number of filters: {0}".format(last_layer_filter))


def get_dogsCats_activations_fromVGG16(source_dir, num_features):
    # Memory for data
    activations = np.zeros((num_features, activation_w, activation_h, last_layer_filter))  # block5_pool shape
    labels = np.zeros((num_features,))

    img_generator = ImageDataGenerator(rescale=1. / 255.)
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
        # vgg16 exit activations
        activations[i * batch_size:(i + 1) * batch_size] = output_batch
        labels[i * batch_size:(i + 1) * batch_size] = label_batch  # labels
        print("Batch: {0}/{1}".format(i, total_batchs))
        i -= 1
        if i == -1:
            break

    return activations, labels


train_data_dir = DIR_TRAIN
validation_data_dir = DIR_VALID

train_activations, train_labels = get_dogsCats_activations_fromVGG16(train_data_dir, ntrain)
validation_activations, validation_labels = get_dogsCats_activations_fromVGG16(validation_data_dir, nvalid)

save('train_activations.npy', train_activations)
save('train_labels.npy', train_labels)
save('validation_activations.npy', validation_activations)
save('validation_labels.npy', validation_labels)
