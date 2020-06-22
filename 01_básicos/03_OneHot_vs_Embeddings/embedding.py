import general
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import Sequential
import tensorflow as tf

d = general.get_data()

model = Sequential()
model.add(Embedding(
                input_dim=10,
                output_dim=3
        ))

model.compile('rmsprop', 'mse')

print(d[0:10])

output_array = model.predict(d)

print(output_array[0:10])

"""
print(d[0:10])

data_one_hot = enc.transform(d[0:10]).toarray()

print(data_one_hot)

a = [data_one_hot[0]]

data_one_hot_distance = [np.linalg.norm(a-b) for b in data_one_hot]
data_one_hot_distance = np.array(data_one_hot_distance)
all = np.insert(data_one_hot, 9, data_one_hot_distance, axis=1)

print(np.array_str(all, max_line_width=np.inf))
"""