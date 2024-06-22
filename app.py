import streamlit as st
import numpy as np
import joblib

from predict import ScalerTransformer, CleanAndTransformation, NeuralNetworkTensorFlow, predict

st.title('Predicciones: WeatherAUS Dataset')

# Cargar los pipelines de predicción 
#C:\Users\arena\OneDrive\FCEIA\TUIA\MLI_1C_2024\AA1-TUIA-Arenas\
pipeline_pred_r = joblib.load(r'pipeline_r.joblib')
pipeline_pred_c = joblib.load(r'pipeline_c.joblib')

# Selección del tipo de predicción
st.sidebar.header('TIPO DE PREDICCION')

prediction_type = st.sidebar.radio(
    "SELECCIONA EL TIPO DE PREDICCION:",
    ('Regresión', 'Clasificación')
)

# Ingreso de las características
st.sidebar.header('VALORES DE LAS FEATURES PARA PREDECIR:')

# Texto descriptivo justo debajo del encabezado
st.sidebar.write('Ignorar las columnas: Unnamed: 0 - RainTomorrow - RainfallTomorrow')

Date = st.sidebar.text_input('Date (AAAA-MM-DD)', value='2024-01-01')

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


MinTemp = st.sidebar.number_input('MinTemp', min_value=-15.0, max_value=50.0, value = 11.0)
MaxTemp = st.sidebar.number_input('MaxTemp', min_value=-15.0, max_value=50.0, value = 21.0)
Rainfall = st.sidebar.number_input('Rainfall', min_value=0.0, max_value=300.0, value = 2.05)
Evaporation = st.sidebar.number_input('Evaporation', min_value=0.0, max_value=100.0, value = 4.83)
Sunshine = st.sidebar.number_input('Sunshine', min_value=0.0, max_value=20.0, value = 6.89)
WindGustDir = st.sidebar.text_input('WindGustDir', value='N')
WindGustSpeed = st.sidebar.number_input('WindGustSpeed', min_value=0.0, max_value=200.0, value = 41.84)
WindDir9am = st.sidebar.text_input('WindDir9am', value='N')
WindDir3pm = st.sidebar.text_input('WindDir3pm', value='N')
WindSpeed9am = st.sidebar.number_input('WindSpeed9am', min_value=0.0, max_value=100.0, value = 15.13)
WindSpeed3pm = st.sidebar.number_input('WindSpeed3pm', min_value=0.0, max_value=100.0, value = 20.02)
Humidity9am = st.sidebar.number_input('Humidity9am', min_value=0.0, max_value=100.0, value = 68.69)
Humidity3pm = st.sidebar.number_input('Humidity3pm', min_value=0.0, max_value=100.0, value = 50.39)
Pressure9am = st.sidebar.number_input('Pressure9am', min_value=0.0, max_value=2000.0, value = 1018.23)
Pressure3pm = st.sidebar.number_input('Pressure3pm', min_value=0.0, max_value=2000.0, value = 1016.12)
Cloud9am = st.sidebar.number_input('Cloud9am', min_value=0.0, max_value=25.0, value = 4.9)
Cloud3pm = st.sidebar.number_input('Cloud3pm', min_value=0.0, max_value=25.0, value = 4.93)
Temp9am = st.sidebar.number_input('Temp9am', min_value=-15.0, max_value=50.0, value = 15.5)
Temp3pm = st.sidebar.number_input('Temp3pm', min_value=-15.0, max_value=50.0, value = 20.43)
RainToday = st.sidebar.text_input('RainToday (Yes/No)', value='Yes')

data_pred = {
            'Date': Date,
            'Location': Location,
            'MinTemp': MinTemp,
            'MaxTemp': MaxTemp,
            'Rainfall': Rainfall,
            'Evaporation': Evaporation,
            'Sunshine': Sunshine,
            'WindGustDir': WindGustDir,
            'WindGustSpeed': WindGustSpeed,
            'WindDir9am': WindDir9am,
            'WindDir3pm': WindDir3pm,
            'WindSpeed9am': WindSpeed9am,
            'WindSpeed3pm': WindSpeed3pm,
            'Humidity9am': Humidity9am,
            'Humidity3pm': Humidity3pm,
            'Pressure9am': Pressure9am,
            'Pressure3pm': Pressure3pm,
            'Cloud9am': Cloud9am,
            'Cloud3pm': Cloud3pm,
            'Temp9am': Temp9am,
            'Temp3pm': Temp3pm,
            'RainToday': RainToday
        }

# Realizar la predicción según el tipo seleccionado

if prediction_type == 'Regresión':
    pred_r = predict(data_pred, prediction_type, pipeline_pred_r = pipeline_pred_r)
    st.write('Predicción | Lluvia estimada para mañana: ', pred_r, 'mm')

else:
    pred_c = predict(data_pred, prediction_type, pipeline_pred_c = pipeline_pred_c)

    # Ajustar el valor a una experiencia de usuario general
    st.write('Predicción | Mañana llueve: ', 'Sí' if pred_c.item() else 'No')

# python -m streamlit run app.py