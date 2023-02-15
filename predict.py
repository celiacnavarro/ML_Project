from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import pickle

with open('my_model', 'rb') as f:
    model = pickle.load(f)

model.train('data/train_arima.csv')
y_pred = model.predict(25)

y_pred.to_csv("predictions.csv", index=False)
