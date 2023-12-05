import streamlit as st
import pandas as pd
import joblib

# Título de la app
st.title('Predicción de lluvia')

# Cargamos el dataset para obtener el nombre
# de las columnas
df = pd.read_csv('datasets/df_reg.csv', index_col=0)

# Cargamos los pipelines
PATH_REG  = 'models/regresion_pipeline.joblib'
PATH_CLAS = 'models/clasificacion_pipeline.joblib'

pipeline_reg  = joblib.load(PATH_REG)
pipeline_clas = joblib.load(PATH_CLAS)

feature_names = pipeline_reg.named_steps['imputer']\
                            .get_feature_names_out()


# Definimos los nombres de las variables
# elimnamos las variable de dirección que 
# no las vamos a usar para la app.
columnas_numericas = list(df.columns[:-2])

cols_dir = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

for col in cols_dir:
    columnas_numericas.remove(col)

# Generamos los sliders para
# cada variable númerica
features = [st.slider(columna,
            df[columna].min(),
            df[columna].max(),
            round(df[columna].mean(), 2)) 
            for columna in columnas_numericas]

# Mapeamos la opción booleana a un texto
# y la agregamos para la predicción junto a
# las variables númericas
raintoday_option_mapping = {'Sí': 1, 'No': 0}
raintoday_option = st.selectbox('Seleccione RainToday:',
                                list(raintoday_option_mapping.keys()))

all_features = features + [raintoday_option_mapping[raintoday_option]]

data_para_predecir = pd.DataFrame([all_features],
                                  columns=feature_names)

# Hacemos las predicciones con el input del front
pred_reg = pipeline_reg.predict(data_para_predecir)
pred_clas = pipeline_clas.predict(data_para_predecir)

# Mostramos las predicciones en la app
st.write('Predicción: ', pred_reg.round(2))
st.write('Predicción: ', pred_clas)