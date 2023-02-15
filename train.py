import pickle
import datetime
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA

with open('my_model', 'rb') as f:
    model = pickle.load(f)

model.train('data/train_arima.csv')

with open(('model_', datetime.datetime.now().strftime('%Y%m%d%H%M%S')), 'wb') as archivo_salida:
    pickle.dump(model, archivo_salida)
