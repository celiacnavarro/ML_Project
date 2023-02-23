#Funciones

def load_data():
    import pandas as pd

    X_train = pd.read_csv('data/regression_X_train.csv', index_col=0)
    X_test = pd.read_csv('data/regression_X_test.csv', index_col=0)
    y_train = pd.read_csv('data/regression_y_train.csv', index_col=0)
    y_test = pd.read_csv('data/regression_y_test.csv', index_col=0)

    return X_train, X_test, y_train, y_test


def df_error(predictions,y_test,filename:str):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


    ''' 
    Crea un dataframe con las métricas de error del modelo
    Args: 
    - Predicciones realizadas con el modelo
    - y_test
    - Nombre del archivo que queremos guardar

    Returns: dataframe
    '''
    error = pd.DataFrame({'Metric': ['MAE', 'MAPE', 'MSE', 'RMSE'], 
                'Error': [mean_absolute_error(y_test,predictions), 
                mean_absolute_percentage_error(y_test,predictions), 
                mean_squared_error(y_test,predictions), 
                np.sqrt(mean_squared_error(y_test,predictions))]})
    
    error.to_csv(filename)
    return error

