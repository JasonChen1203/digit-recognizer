import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataprocessing import *




# define CNN model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# train the model
def train_model(dataX, dataY, n_epochs, batch_size):
    #define model
    model = define_model()

     #fit model
    model.fit(dataX, dataY, epochs=n_epochs, batch_size=batch_size, verbose=0)

    return model


# evaluate model using k-fold cross validation
def evaluate_model(dataX, dataY, n_epochs, batch_size, k_folds=3):
    kf = KFold(k_folds, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(dataX):
        # define evaluation data
        trainX, trainY, testX, testY = dataX[train_index], dataY[train_index], dataX[test_index], dataY[test_index]

        # train model based on given data
        current_model = train_model(trainX, trainY, n_epochs, batch_size)

        # evaluate model
        _, acc = current_model.evaluate(testX, testY)

        # return scores
        print('> Model Accuracy: %.3f' % (acc * 100.0))


# summarize model preformance
def diagnose_model(n_epochs, batch_size):
    # load dataset
    trainX, trainY, testX, testY = load_data()

    # evaluate model
    evaluate_model(trainX, trainY, n_epochs, batch_size)


# save model
def save_model(n_epochs, batch_size):
    trainX, trainY, testX, testY = load_data()

    model = train_model(trainX, trainY, n_epochs, batch_size)
    
    model.save("src/Model/final_model")



save_model(30, 32)
