import datetime
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from utils import functions
from keras.models import load_model, save_model

import pickle 
with open('my_model_regression', 'rb') as f:
    model = pickle.load(f)

X_train, X_test, y_train, y_test = functions.load_data()


model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_df =pd.DataFrame(y_pred, columns=['predictions']) 

a = 'data/predictions/regression_predictions_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
y_pred_df.to_csv(a, index=False)
