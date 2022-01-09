import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Model.dataprocessing import *


# Processing test dataset from Kaggle and return the predictions in csv format
 
# load kaggle test dataset
def load_kaggle_data():
  df_test = pd.read_csv("src/KaggleMaterial/KaggleData/test.csv")

  testX = df_test.to_numpy()
  testX = testX.reshape((testX.shape[0], 28, 28, 1))

  test_norm = testX.astype('float32')
  test_norm /= 255.0

  return test_norm


# feed kaggle dataset into model and return model predictions
def make_kaggle_prediction():
    # Load prebuilt model
    model = keras.models.load_model('src/Model/final_model')
    
    testX = load_kaggle_data()
    predictions = model.predict(testX)

    return predictions


# save predictions in csv file
def make_kaggle_submission():
    predictions = make_kaggle_prediction()
    df_imageid = []
    df_predictions = []

    for i in range(1, len(predictions)+1):
        df_imageid.append(i)

    for prediction in predictions:
        df_predictions.append(np.argmax(prediction))

    df_submission = pd.DataFrame({'ImageId': df_imageid, 'Label':df_predictions})
    df_submission.to_csv("src/KaggleMaterial/KaggleSubmission/kaggle_submission.csv")


make_kaggle_submission()

