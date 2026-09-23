import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------
# Cargar modelo y archivos de apoyo
# ---------------------------------------------------------
modelo = joblib.load('modelo_congestion_transmilenio.pkl')
columnas_modelo = joblib.load('columnas_modelo.pkl')
mapeo_estaciones = joblib.load('mapeo_estaciones.pkl')
umbrales = joblib.load('umbrales_congestion.pkl')
resumen_dia_hora = joblib.load('resumen_dia_hora.pkl')  # index=hora, columnas=día, valores=validaciones

st.set_page_config(page_title="TransMilenio - Congestión", page_icon="🚌", layout="centered")

# ---------------------------------------------------------
# Estilo (tarjetas redondeadas, look cálido)
# ---------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
h1 { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚌 ¿Qué tan lleno va a estar?</h1>", unsafe_allow_html=True)
st.caption(
    "Estimación basada en un **proxy de congestión** (validaciones esperadas por hora), "
    "no en un tiempo de espera medido directamente. Datos: 4-17 mayo 2026."
)

orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
dias_habiles = orden_dias[:5]

# ---------------------------------------------------------
# Entradas del usuario
# ---------------------------------------------------------
dia_semana = st.segmented_control("📅 Día de la semana", orden_dias, default="Miércoles")
if dia_semana is None:
    dia_semana = "Miércoles"

nombre_estacion = st.selectbox("📍 Estación", mapeo_estaciones['nombre_estacion'].tolist())
codigo_estacion = mapeo_estaciones.loc[
    mapeo_estaciones['nombre_estacion'] == nombre_estacion, 'codigo_estacion'
].values[0]

hora = st.slider("🕐 Hora de salida", 0, 23, 7)

# ---------------------------------------------------------
# Calcular hora_pico_real (misma lógica del entrenamiento)
# ---------------------------------------------------------
en_pico_manana = 6 <= hora <= 8
en_pico_tarde = 16 <= hora <= 18
hora_pico_real = 'Pico' if (dia_semana in dias_habiles and (en_pico_manana or en_pico_tarde)) else 'No pico'

# ---------------------------------------------------------
# Construir fila de entrada (mismo formato de X del entrenamiento)
# ---------------------------------------------------------
fila = pd.DataFrame(0, index=[0], columns=columnas_modelo)
fila['hora'] = hora
for col in [f'codigo_estacion_{codigo_estacion}', f'dia_semana_{dia_semana}', f'hora_pico_real_{hora_pico_real}']:
    if col in fila.columns:
        fila[col] = 1

# ---------------------------------------------------------
# Predicción → nivel interpretable
# ---------------------------------------------------------
prediccion = modelo.predict(fila)[0]

if prediccion <= umbrales['q33']:
    nivel, emoji, color_fondo = "Baja", "🟢", "#DFF5E1"
    mensaje = "Buen momento para viajar — poca afluencia esperada."
elif prediccion <= umbrales['q66']:
    nivel, emoji, color_fondo = "Media", "🟡", "#FFF6D9"
    mensaje = "Afluencia moderada — espera algo de gente, sin aglomeración fuerte."
else:
    nivel, emoji, color_fondo = "Alta", "🔴", "#FBE1E1"
    mensaje = "Momento de alta afluencia — espera estaciones llenas y posible espera adicional."

st.markdown(f"""
<div style='background-color:{color_fondo}; padding:28px; border-radius:20px; text-align:center; margin: 20px 0;'>
    <div style='font-size:44px;'>{emoji}</div>
    <div style='font-size:26px; font-weight:700; color:#1a1a1a;'>Congestión {nivel}</div>
    <div style='font-size:15px; margin-top:8px; color:#333;'>{mensaje}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Gráfica 1: patrón por hora del día seleccionado vs promedio
# ---------------------------------------------------------
st.subheader(f"📊 ¿Cómo se mueve un {dia_semana.lower()}, hora por hora?")

promedio_horas = resumen_dia_hora.mean(axis=1)
serie_dia = resumen_dia_hora[dia_semana]
hora_pico_dia = int(serie_dia.idxmax())

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(promedio_horas.index, promedio_horas.values, linestyle='--', color='gray', label='Promedio de todos los días')
ax1.plot(serie_dia.index, serie_dia.values, color='#2c7fb8', linewidth=2.5, label=dia_semana)
ax1.axvline(hora, color='red', linestyle=':', label='Tu hora seleccionada')
ax1.set_xlabel('Hora del día')
ax1.set_ylabel('Validaciones (todas las estaciones)')
ax1.legend()
st.pyplot(fig1)

diferencia_pct = ((serie_dia.sum() - promedio_horas.sum()) / promedio_horas.sum()) * 100
if diferencia_pct > 5:
    texto_dif = f"tiene un {diferencia_pct:.0f}% MÁS de validaciones que el día promedio"
elif diferencia_pct < -5:
    texto_dif = f"tiene un {abs(diferencia_pct):.0f}% MENOS de validaciones que el día promedio"
else:
    texto_dif = "se comporta muy similar al promedio del sistema"

st.caption(f"🕐 Hora más congestionada de un {dia_semana.lower()}: {hora_pico_dia}:00 h. Este día {texto_dif}.")

# ---------------------------------------------------------
# Gráfica 2: comparación entre los 7 días
# ---------------------------------------------------------
st.subheader("📅 ¿Qué día tiene más movimiento en total?")

totales_por_dia = resumen_dia_hora.sum(axis=0)[orden_dias]
colores = ['#2c7fb8' if d == dia_semana else '#d9d9d9' for d in orden_dias]

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.bar(totales_por_dia.index, totales_por_dia.values, color=colores)
ax2.set_ylabel('Total de validaciones')
plt.xticks(rotation=20)
st.pyplot(fig2)

ranking = totales_por_dia.rank(ascending=False).astype(int)
posicion = ranking[dia_semana]
dia_top = totales_por_dia.idxmax()
st.caption(
    f"El {dia_semana.lower()} ocupa el puesto #{posicion} de 7 en volumen total. "
    f"El día con más movimiento en todo el periodo fue **{dia_top}**."
)

# ---------------------------------------------------------
# Detalles técnicos (opcional, no estorba al usuario final)
# ---------------------------------------------------------
with st.expander("🔍 Ver detalles técnicos del modelo"):
    st.write(f"Validaciones/hora estimadas usadas para calcular el nivel de arriba: **{prediccion:,.0f}**")
    st.write("Variables que más influyen en las predicciones del modelo:")

    importancias = pd.Series(modelo.feature_importances_, index=columnas_modelo)
    top5 = importancias.sort_values(ascending=False).head(5)
    st.bar_chart(top5)

    st.caption(
        "Nota metodológica: el dataset original no incluye un campo confiable de hora pico "
        "ni de tiempo de espera real; ambos se derivaron a partir de la hora de cada validación."
    )
