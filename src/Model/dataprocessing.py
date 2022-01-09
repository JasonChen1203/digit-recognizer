import pandas as pd
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# load mnist dataset and reshape them into appropriate formt to feed into model


# load dataset from mnist
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()

    # reshape dataset
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # one hot encoding
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


# prepare data to feed into model
def prep_data(dataX):
    dataX_norm = dataX.astype('float32')
    dataX_norm /= 255.0

    return dataX_norm


# load dataset from mnist and scale them to feed into model
def load_data():
    trainX, trainY, testX, testY = load_dataset()

    trainX = prep_data(trainX)
    testX = prep_data(testX)

    return trainX, trainY, testX, testY
