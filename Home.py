import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Predicción de Asistencia a Museos",
    page_icon="📊",
    layout="wide"
)

st.title("Sistema de Predicción de Asistencia a Museos")

st.markdown("""
Esta aplicación representa el despliegue productivo de la solución predictiva desarrollada
para estimar la asistencia mensual a museos a partir de datos históricos.

El modelo seleccionado para la predicción en vivo es **XGBoost sin optimizar**, debido a que
obtuvo el mejor desempeño global en la comparación final de modelos.
""")

st.subheader("Resumen del proyecto")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Periodo trabajado", "2007 - 2026")

with col2:
    st.metric("Modelo final", "XGBoost")

with col3:
    st.metric("MAPE", "17.21%")

st.subheader("Vistas disponibles")

st.write("""
Desde el menú lateral podrás acceder a las siguientes secciones:

1. Análisis histórico de asistencia.
2. Predicción en vivo con nuevo registro.
3. Comparación de modelos.
4. Conclusiones del proyecto.
""")
