from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import pickle
import pandas as pd

with open('my_model', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/train_arima.csv')
model.fit(df)

y_pred = model.predict(25)

y_pred.to_csv("predictions.csv", index=False)
