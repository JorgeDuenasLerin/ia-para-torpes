import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

labeled_ds, summary = tfds.load('oxford_flowers102', split='train+test+validation', with_info=True, as_supervised=True)

e=next(iter(labeled_ds.shuffle(4)))

array_image = e[0]
value_label = e[1]
label = tfds.image_classification.oxford_flowers102._NAMES[value_label]


plt.imshow(array_image, interpolation='nearest')
plt.title(label)
plt.show()