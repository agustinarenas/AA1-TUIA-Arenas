from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NeuralNetworkTensorFlow:
    """
        Modelo flexible con TensorFlow para resolver problemas de regresión o clasificación.
        Permite personalizar la cantidad de capas, neuronas por capa, funciones de activación, etc.
    """
    def __init__(self, input_shape, num_layers=2, neurons_per_layer=2, activation='relu', 
                 output_activation='linear', num_output_neurons=1, loss='mean_squared_error', 
                 optimizer='adam', learning_rate=0.01, epochs=100, batch_size=99, metrics=None, cant_params=True):
        
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.output_activation = output_activation
        self.num_output_neurons = num_output_neurons
        self.loss = loss
        self.learning_rate = learning_rate
        self.optimizer = self.get_optimizer(optimizer, learning_rate)
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = metrics if metrics is not None else []
        self.cant_params = cant_params
        self.model = self.build_model()


    def get_optimizer(self, optimizer, learning_rate):
        """
        Devuelve una instancia del optimizador con la tasa de aprendizaje especificada.
        """
        optimizers = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adagrad': tf.keras.optimizers.Adagrad,
        }
        if optimizer in optimizers:
            return optimizers[optimizer](learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer '{optimizer}' no es válido.")

    def build_model(self):
        """
            Construye el modelo con las especificaciones dadas.
        """
        model = tf.keras.Sequential()
        
        # Agregar la primera capa con input_shape
        model.add(tf.keras.layers.Dense(self.neurons_per_layer, activation=self.activation, input_shape=(self.input_shape,)))
        
        # Agregar capas ocultas adicionales
        for _ in range(self.num_layers - 1):
            model.add(tf.keras.layers.Dense(self.neurons_per_layer, activation=self.activation))
        
        # Agregar la capa de salida
        model.add(tf.keras.layers.Dense(self.num_output_neurons, activation=self.output_activation))
        
        # Compilar el modelo
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        if self.cant_params:
            # Imprimir la cantidad de parámetros a modo de ejemplo
            print("n° de parámetros:", model.count_params())

        return model
    
    def fit(self, X, y):
        """
            Entrena el modelo.
        """
        X = np.array(X)
        y = np.array(y)
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return history.history['loss']

    def evaluate(self, X, y):
        """
            Evalúa el modelo.
        """
        X = np.array(X)
        y = np.array(y)
        loss = self.model.evaluate(X, y, verbose=0)
        return loss

    def predict(self, X):
        """
            Hace predicciones con el modelo entrenado.
        """
        X = np.array(X)
        predictions = self.model.predict(X)
        
        # Convertir probabilidades a valores binarios si la activación de salida es sigmoide
        if self.output_activation == 'sigmoid':
            predictions = (predictions >= 0.5).astype(int)
        
        return predictions
    
    def get_params(self, deep=True):
        return {
            'input_shape': self.input_shape,
            'num_layers': self.num_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'num_output_neurons': self.num_output_neurons,
            'loss': self.loss,
            'optimizer': self.optimizer_name,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'metrics': self.metrics,
            'cant_params': self.cant_params
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        # Recompilar el modelo si se cambian los hiperparámetros
        self.model = self.build_model()
        return self

class CleanAndTransformation(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.unique_categories_ = None

    def fit(self, dfx):
        # Almacenar las categorías únicas para cada columna categórica
        self.unique_categories_ = {
            'Location': np.sort(np.array(['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne',
                                            'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport'])),
                                
            'month': np.sort(np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])),

            'WindGustDir': np.sort(np.array(['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
                                                'SSW', 'SW', 'W', 'WNW', 'WSW'])),

            'WindDir9am': np.sort(np.array(['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
                                                'SSW', 'SW', 'W', 'WNW', 'WSW'])),
        
            'WindDir3pm': np.sort(np.array(['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
                                                'SSW', 'SW', 'W', 'WNW', 'WSW'])),
        }
        return self

    def transform(self, dfx):
        # Transformación del diccionario del Frontend a un DataFrame para su tratamiento
        if isinstance(dfx, dict):
            # Convertir el diccionario en un DataFrame
            dfx = pd.DataFrame(dfx, index=[0])

        # Date a mont: Nos quedamos con la información del mes de la columna Date
        dfx['month'] = dfx.Date.apply(lambda x: str(x).split("-")[1])

        dfx = dfx.drop("Date", axis=1)

        # Modificamos la columna booleana para que se vea en la forma numérica
        dfx.RainToday = dfx.RainToday.map({'No': 0, 'Yes': 1})

        # Operaciones de columnas 
        dfx['win_prom'] = (dfx.WindSpeed9am + dfx.WindSpeed3pm)/2

        dfx['hum_prom'] = (dfx.Humidity9am + dfx.Humidity3pm)/2

        dfx['clo_prom'] = (dfx.Cloud9am + dfx.Cloud3pm)/2

        dfx['pres_delta'] = dfx.Pressure9am - dfx.Pressure3pm

        dfx['tem_delta'] = dfx.Temp9am - dfx.Temp3pm

        # Dummy
        categorias = ['Location', 'month', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
        
        # Crear todas las columnas dummy necesarias
        for cat in categorias:
            for val in self.unique_categories_[cat]:
                dfx[f"dy_{cat}_{val}"] = 0

        # Convertir las columnas especificadas en variables dummy
        for cat in categorias:
            col_prefix = f"dy_{cat}_"
            for val in self.unique_categories_[cat]:
                if val in dfx[cat].values:
                    dfx.loc[dfx[cat] == val, f"{col_prefix}{val}"] = 1

        # Eliminar las columnas categóricas originales
        dfx = dfx.drop(categorias, axis=1)

        return dfx

class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_scale = [
        'MinTemp',
        'MaxTemp',
        'Rainfall',
        'Evaporation',
        'Sunshine',
        'WindGustSpeed',
        'WindSpeed9am',
        'WindSpeed3pm',
        'Humidity9am',
        'Humidity3pm',
        'Pressure9am',
        'Pressure3pm',
        'Cloud9am',
        'Cloud3pm',
        'Temp9am',
        'Temp3pm',
        'win_prom',
        'hum_prom',
        'clo_prom',
        'pres_delta',
        'tem_delta'
        ]
        self.media = 0
        self.desvio_estandar = 1

    def fit(self, dfx):
        
        # Medias y desviaciones estándar
        medias = {
        'MinTemp': 11.306170,
        'MaxTemp': 21.933696,
        'Rainfall': 1.992721,
        'Evaporation': 4.774644,
        'Sunshine': 7.369457,
        'WindGustSpeed': 41.737005,
        'WindSpeed9am': 15.088307,
        'WindSpeed3pm': 20.011445,
        'Humidity9am': 68.697091,
        'Humidity3pm': 50.397304,
        'Pressure9am': 1018.236301,
        'Pressure3pm': 1016.129651,
        'Cloud9am': 4.909878,
        'Cloud3pm': 4.936155,
        'Temp9am': 15.500263,
        'Temp3pm': 20.430236,
        'win_prom': 17.552765,
        'hum_prom': 59.542958,
        'clo_prom': 4.780495,
        'pres_delta': 2.106527,
        'tem_delta': -4.928334
        }

        desvios_estandar = {
        'MinTemp': 5.692168,
        'MaxTemp': 6.680122,
        'Rainfall': 6.316537,
        'Evaporation': 3.604916,
        'Sunshine': 3.696438,
        'WindGustSpeed': 13.883248,
        'WindSpeed9am': 9.464180,
        'WindSpeed3pm': 8.919395,
        'Humidity9am': 18.739591,
        'Humidity3pm': 19.916131,
        'Pressure9am': 7.328217,
        'Pressure3pm': 7.175800,
        'Cloud9am': 2.425839,
        'Cloud3pm': 2.226711,
        'Temp9am': 5.686474,
        'Temp3pm': 6.518882,
        'win_prom': 8.017439,
        'hum_prom': 17.687629,
        'clo_prom': 1.973002,
        'pres_delta': 2.074142,
        'tem_delta': 3.424439
        }

        # Convertir a Series
        self.media = pd.Series(medias)
        self.desvio_estandar = pd.Series(desvios_estandar)

        return self

    def transform(self, dfx):
        dfx_std = dfx.copy()
        
        # Estandarizar el DataFrame
        dfx_std[self.cols_scale] = (dfx_std[self.cols_scale] - self.media) / self.desvio_estandar

        return dfx_std
    
def predict(data_pred, prediction_type, pipeline_pred_r  = None, pipeline_pred_c = None):
    if prediction_type == 'Regresión':
        # Aplicamos pipeline_pred_r
        data_ct = pipeline_pred_r['Clean and Transformation'].transform(data_pred)
        data_std = pipeline_pred_r['Standard Scaler'].transform(data_ct)
        pred_r = pipeline_pred_r['Model'].predict(data_std)

        # Ajustar el valor a cero si es negativo
        pred_r = max(round(pred_r.item(), 2), 0)

        return pred_r

    else:
        # Aplicamos pipeline_pred_c
        data_ct = pipeline_pred_c['Clean and Transformation'].transform(data_pred)
        data_std = pipeline_pred_c['Standard Scaler'].transform(data_ct)
        pred_c = pipeline_pred_c['Model'].predict(data_std)

        return pred_c
