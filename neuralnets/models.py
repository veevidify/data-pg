import numpy as np

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

def linear_regression(X, y):
    model = Sequential()
    model.add(Dense(1, input_shape=(2,), activation='linear'))

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='mse', optimizer=sgd)

    model.fit(X, y, epochs=100, batch_size=2)
    return model
