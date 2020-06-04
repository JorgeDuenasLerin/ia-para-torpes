"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from os.path import dirname, join as pjoin
"""

import scipy.io as sio
import json
import pprint

dict_mat_contents = sio.loadmat('../data/setid.mat')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(dict_mat_contents)