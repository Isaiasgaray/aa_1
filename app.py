import streamlit as st
import pandas as pd
import joblib

# 1. Configuración inicial de la página (¡Debe ser la primera línea de Streamlit!)
st.set_page_config(page_title="Pronóstico de Lluvia", page_icon="🌤️", layout="centered")

# Título y descripción
st.title('🌤️ Pronóstico de lluvia para mañana')
st.markdown("Ajustá las condiciones meteorológicas actuales para predecir el clima de mañana en Australia.")
st.divider() # Línea separadora visual

# Cargamos el archivo (con la lógica robusta)
df_crudo = pd.read_csv('datasets/df_clima_australia.csv')

PATH_REG  = 'models/regresion_pipeline.joblib'
PATH_CLAS = 'models/clasificacion_pipeline.joblib'

pipeline_reg  = joblib.load(PATH_REG)
pipeline_clas = joblib.load(PATH_CLAS)

feature_names = pipeline_reg.named_steps['imputer'].get_feature_names_out()
columnas_numericas = [col for col in feature_names if col != 'RainToday']

# Calculamos medias y desvíos matemáticos (ocultos al usuario)
means = {}
stds = {}
for col in columnas_numericas:
    col_limpia = pd.to_numeric(df_crudo[col], errors='coerce')
    if col_limpia.isna().all():
        means[col] = 0.0
        stds[col] = 1.0
    else:
        means[col] = col_limpia.mean()
        stds[col] = col_limpia.std(ddof=0)

input_data = {}

# 2. Sidebar (Panel lateral) para opciones generales
with st.sidebar:
    st.header("⚙️ Configuración inicial")
    raintoday_option = st.selectbox('¿Hoy llovió?', ['Sí', 'No'])
    input_data['RainToday'] = 1 if raintoday_option == 'Sí' else 0
    st.info("Deslizá los valores en la pantalla principal para ver cómo cambia la predicción.")

# 3. Dividimos la pantalla en 2 columnas para que los sliders no queden tan largos
col1, col2 = st.columns(2)

# Repartimos los sliders dinámicamente entre las dos columnas
for i, col in enumerate(columnas_numericas):
    col_limpia = pd.to_numeric(df_crudo[col], errors='coerce')
    min_val = float(col_limpia.min()) if not col_limpia.isna().all() else 0.0
    max_val = float(col_limpia.max()) if not col_limpia.isna().all() else 360.0
    mean_val = float(means[col])
    
    if min_val == max_val:
        min_val, max_val = 0.0, max_val + 1.0
        
    # Si el índice es par va a la columna 1, si es impar va a la 2
    if i % 2 == 0:
        with col1:
            valor_real = st.slider(col, min_val, max_val, mean_val)
    else:
        with col2:
            valor_real = st.slider(col, min_val, max_val, mean_val)
    
    # Normalización matemática
    std_divisor = stds[col] if stds[col] != 0 else 1.0
    input_data[col] = (valor_real - means[col]) / std_divisor

# Predicciones
data_para_predecir = pd.DataFrame([input_data], columns=feature_names)
pred_clas = pipeline_clas.predict(data_para_predecir)
pred_reg = pipeline_reg.predict(data_para_predecir)

resultado_reg  = round(float(pred_reg[0][0] if hasattr(pred_reg[0], '__iter__') else pred_reg[0]), 2)
if resultado_reg < 0:
    resultado_reg = 0.0

# 4. Sección de Resultados Visuales
st.divider()
st.subheader("🎯 Resultado de la IA")

# 5. Tarjetas de métricas y alertas de color dependiendo del resultado
if pred_clas[0]: # Si llueve
    st.warning("⚠️ **Pronóstico:** Alta probabilidad de lluvia para mañana.")
    st.metric(label="🌧️ Cantidad de lluvia estimada", value=f"{resultado_reg} mm")
else: # Si no llueve
    st.success("✅ **Pronóstico:** Día despejado. No se esperan lluvias.")
    st.metric(label="🌞 Cantidad de lluvia estimada", value="0 mm")