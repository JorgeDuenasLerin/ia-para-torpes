import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt





#tr, ts, va = tfds.load('oxford_flowers102', split=['train','test','validation'])

tr = tfds.load('oxford_flowers102', split='train[0:2]')


def extract_img_val_to_array(e):
    print (type(e['image']))
    return e['image']
    #return i.numpy().flatten()


def extract_label_val(e):
    return e['label']


X_train = tr.map(lambda e: extract_img_val_to_array(e))
y_train = tr.map(lambda e: extract_label_val(e))
X_train = tfds.as_numpy(X_train)
y_train = tfds.as_numpy(y_train)


for X, y in X_train, y_train:
    print(X)
    print(y)