import streamlit as st
import numpy as np
import joblib

st.title('Predicciones: WeatherAUS Dataset')

# Cargar los pipelines de predicción
pipeline_pred_r = joblib.load('pipeline_r.joblib')
pipeline_pred_c = joblib.load('pipeline_c.joblib')

# Ingreso de las características
st.sidebar.header('Valores de las features para predecir:')
'''Ignorar las columnas Unnamed_0 - RainTomorrow - RainFallTomorrow''' # Unnamed_0 = st.sidebar.number_input('Unnamed_0', value=0.0)

Date = st.sidebar.text_input('Date')

# Lista de ubicaciones permitidas
ciudades_permitidas = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport']
ciudades_permitidas.append("Otra")  # Añadir opción para ingresar manualmente

# Crear un menú desplegable (selectbox) en la barra lateral
selected_location = st.sidebar.selectbox('Seleccionar localidad o Ingresar localidad', ciudades_permitidas)

# Permitir al usuario escribir una ubicación si selecciona "Otra"
if selected_location == "Otra":
    location_input = st.sidebar.text_input('Ingresar localidad')
    if location_input:
        Location = location_input
    else:
        st.sidebar.error("Por favor ingrese una localidad valida")
else:
    Location = selected_location


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
RainToday = st.sidebar.text_input('RainToday (Yes/No)')

# Selección del tipo de predicción
prediction_type = st.sidebar.radio(
    "Selecciona el tipo de predicción:",
    ('Regresión', 'Clasificación')
)

# Crear el array de datos para la predicción
data_pred = np.array([[Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, 
                       WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday]])

# Realizar la predicción según el tipo seleccionado
if prediction_type == 'Regresión':
    pred_r = pipeline_pred_r.predict(data_pred)
    st.write('Predicción | Lluvia estimada para mañana: ', pred_r[0], 'mm') # Ojo aca depende de como este preparado el metodo predict de la pipeline puede ser un int
else:
    pred_c = pipeline_pred_c.predict(data_pred)
    st.write('Predicción | Mañana llueve: ', 'Sí' if pred_c[0] else 'No') # Ojo aca depende de como este preparado el metodo predict de la pipeline puede ser un str
