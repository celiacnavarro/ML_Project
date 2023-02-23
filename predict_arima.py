from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import pickle
import pandas as pd
import datetime

with open('my_model_arima', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/train_arima.csv')
model.fit(df)

y_pred = model.predict(25)

a = 'data/predictions/arima_predictions_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv'
y_pred.to_csv(a, index=False)
