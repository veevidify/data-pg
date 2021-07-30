import numpy as np

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

def linear_regression(X, y):
    model = Sequential()
    model.add(Dense(1, input_shape=(X.shape[1],), activation='linear'))

    sgd = optimizers.SGD(lr=0.1)
    model.compile(loss='mse', optimizer=sgd)

    model.fit(X, y, epochs=100, batch_size=2)
    return model

def logistic_regression(X, y):
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    model.add(Activation('sigmoid'))

    sgd = optimizers.SGD(lr=0.05)
    model.compile(loss=binary_crossentropy, optimizer=sgd)

    model.fit(X, y, epochs=1000, batch_size=1)
    return model
