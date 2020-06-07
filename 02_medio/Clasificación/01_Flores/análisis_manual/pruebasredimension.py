import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt


# Dibuja un tf.Tensor

def draw_array(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()


def draw_tensor(tsimg, label):
    draw_array(np.array(tsimg[label]))


def getfirst(ds):
    e = next(iter(ds))
    draw_tensor(e, 'image')



#tr, ts, va = tfds.load('oxford_flowers102', split=['train','test','validation'])

tr = tfds.load('oxford_flowers102', split='train[0:2]')


for t in tr:
    draw_tensor(t, 'image')

"""

Código para jugar:


e = next(iter(tr))
>> draw_array(np.array(io))  
>>> ir = tf.image.resize(io, [100, 100])
>>> irc = tf.cast(ir,dtype=np.uint8)
>>> draw_array(np.array(irc))

"""


def resize_tensor_image(t, label, size):
    new = tf.image.resize(t[label], size)
    new = tf.cast(new, dtype=np.uint8)
    t[label] = new
    return t


tr = tr.map(lambda e: resize_tensor_image(e, 'image', [50, 50]))

for t in tr:
    draw_tensor(t, 'image')
