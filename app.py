import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. Cargar el modelo y los archivos de apoyo
#    (deben estar en la misma carpeta que este app.py)
# ---------------------------------------------------------
modelo = joblib.load('modelo_congestion_transmilenio.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')
mapeo_estaciones = joblib.load('mapeo_estaciones.pkl')

st.set_page_config(page_title="Congestión TransMilenio", page_icon="🚌")
st.title("🚌 Predictor de congestión — TransMilenio")
st.caption(
    "Estimación basada en el histórico de validaciones (4-17 mayo 2026). "
    "El número mostrado es un proxy de congestión (validaciones esperadas), no un tiempo de espera medido directamente."
)

# ---------------------------------------------------------
# 2. Entradas del usuario
# ---------------------------------------------------------
dias_habiles = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']

col1, col2 = st.columns(2)
with col1:
    dia_semana = st.selectbox(
        "Día de la semana",
        ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    )
with col2:
    hora = st.slider("Hora de salida", min_value=0, max_value=23, value=7)

nombre_estacion = st.selectbox(
    "Estación",
    mapeo_estaciones['nombre_estacion'].tolist()
)
codigo_estacion = mapeo_estaciones.loc[
    mapeo_estaciones['nombre_estacion'] == nombre_estacion, 'codigo_estacion'
].values[0]

# ---------------------------------------------------------
# 3. Calcular hora_pico_real con la MISMA lógica usada en Colab
#    (debe coincidir exactamente con cómo se entrenó el modelo)
# ---------------------------------------------------------
en_pico_manana = 6 <= hora <= 8
en_pico_tarde = 16 <= hora <= 18  # simplificado a nivel de hora completa
hora_pico_real = 'Pico' if (dia_semana in dias_habiles and (en_pico_manana or en_pico_tarde)) else 'No pico'

# ---------------------------------------------------------
# 4. Construir la fila de entrada con el mismo formato
#    de one-hot encoding que usó X en el entrenamiento
# ---------------------------------------------------------
fila = pd.DataFrame(0, index=[0], columns=columnas_modelo)
fila['hora'] = hora

col_estacion = f'codigo_estacion_{codigo_estacion}'
col_dia = f'dia_semana_{dia_semana}'
col_pico = f'hora_pico_real_{hora_pico_real}'

for col in [col_estacion, col_dia, col_pico]:
    if col in fila.columns:
        fila[col] = 1

# ---------------------------------------------------------
# 5. Predicción
# ---------------------------------------------------------
prediccion = modelo.predict(fila)[0]

st.metric("Congestión estimada (validaciones/hora, proxy)", f"{prediccion:,.0f}")

if hora_pico_real == 'Pico':
    st.warning(f"Hora pico ({dia_semana}): congestión por encima del promedio del sistema.")
else:
    st.success(f"Fuera de hora pico ({dia_semana}): congestión cercana o por debajo del promedio.")

# ---------------------------------------------------------
# 6. Importancia de variables (igual que en el mockup, con datos reales)
# ---------------------------------------------------------
st.subheader("¿Qué tanto pesa cada variable en el modelo?")

importancias = pd.Series(modelo.feature_importances_, index=columnas_modelo)
top5 = importancias.sort_values(ascending=False).head(5)

st.bar_chart(top5)

st.caption(
    "Nota metodológica: el dataset original no incluye un campo confiable de hora pico "
    "ni de tiempo de espera real; ambos se derivaron a partir de la hora de cada validación."
)
