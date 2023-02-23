import pickle
import datetime
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA
import pandas as pd

with open('my_model_arima', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/train_arima.csv')
model.fit(df)

a = 'model/model_ARIMA_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
with open(a, 'wb') as archivo_salida:
    pickle.dump(model, archivo_salida)
