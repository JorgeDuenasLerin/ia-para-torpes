import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

tr, ts, va = tfds.load('oxford_flowers102', split=['train','test','validation'])