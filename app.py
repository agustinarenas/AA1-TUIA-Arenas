import streamlit as st
import numpy as np
import joblib

st.title('Predicciones: WeatherAUS Dataset')

# Cargar los pipelines de predicción
pipeline_pred_r = joblib.load('pipeline_r.joblib')
pipeline_pred_c = joblib.load('pipeline_c.joblib')

st.sidebar.header('Valores de las features para predecir:')
'''Ignorar las columnas Unnamed_0 - '''
# Ingreso de las características
# Unnamed_0 = st.sidebar.number_input('Unnamed_0', value=0.0)
Date = st.sidebar.text_input('Date')
Location = st.sidebar.text_input('Location')
MinTemp = st.sidebar.number_input('MinTemp', min_value=-15.0, max_value=50.0)
MaxTemp = st.sidebar.number_input('MaxTemp', min_value=-15, max_value=50.0)
Rainfall = st.sidebar.number_input('Rainfall', min_value=0.0, max_value=300.0)
Evaporation = st.sidebar.number_input('Evaporation', min_value=0.0, max_value=100.0)
Sunshine = st.sidebar.number_input('Sunshine', min_value=0.0, max_value=20.0)
WindGustDir = st.sidebar.text_input('WindGustDir')
WindGustSpeed = st.sidebar.number_input('WindGustSpeed', min_value=0.0, max_value=200.0)
WindDir9am = st.sidebar.text_input('WindDir9am')
WindDir3pm = st.sidebar.text_input('WindDir3pm')
WindSpeed9am = st.sidebar.number_input('WindSpeed9am', min_value=0.0, max_value=100.0)
WindSpeed3pm = st.sidebar.number_input('WindSpeed3pm', min_value=0.0, max_value=100.0)
Humidity9am = st.sidebar.number_input('Humidity9am', min_value=0.0, max_value=100.0)
Humidity3pm = st.sidebar.number_input('Humidity3pm', min_value=0.0, max_value=100.0)
Pressure9am = st.sidebar.number_input('Pressure9am', min_value=0.0, max_value=2000.0)
Pressure3pm = st.sidebar.number_input('Pressure3pm', min_value=0.0, max_value=2000.0)
Cloud9am = st.sidebar.number_input('Cloud9am', min_value=0.0, max_value=25.0)
Cloud3pm = st.sidebar.number_input('Cloud3pm', min_value=0.0, max_value=25.0)
Temp9am = st.sidebar.number_input('Temp9am', min_value=-15.0, max_value=50.0)
Temp3pm = st.sidebar.number_input('Temp3pm', min_value=-15.0, max_value=50.0)
RainToday = st.sidebar.number_input('RainToday', min_value=0.0, max_value=1.0)

# Selección del tipo de predicción
prediction_type = st.sidebar.radio(
    "Selecciona el tipo de predicción:",
    ('Regresión', 'Clasificación')
)

# Crear el array de datos para la predicción
data_pred = np.array([[Unnamed_0, Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, 
                       WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday]])

# Realizar la predicción según el tipo seleccionado
if prediction_type == 'Regresión':
    pred_r = pipeline_pred_r.predict(data_pred)
    st.write('Predicción | Lluvia estimada para mañana: ', pred_r[0], 'mm') # Ojo aca depende de como este preparado el metodo predict de la pipeline puede ser un int
else:
    pred_c = pipeline_pred_c.predict(data_pred)
    st.write('Predicción | Mañana llueve: ', 'Sí' if pred_c[0] else 'No') # Ojo aca depende de como este preparado el metodo predict de la pipeline puede ser un str
