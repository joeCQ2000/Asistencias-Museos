import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.title("Comparación de modelos predictivos")

st.write("""
Esta vista presenta la comparación de los modelos evaluados durante el desarrollo del proyecto.
El objetivo es identificar qué modelo obtuvo el mejor desempeño para la predicción mensual de
asistencia a museos.
""")

COMPARACION_PATH = Path("data/comparacion_modelos.csv")
PREDICCIONES_PATH = Path("data/predicciones_xgboost.csv")
RESIDUOS_PATH = Path("data/residuos_xgboost.csv")


@st.cache_data
def cargar_comparacion():
    df = pd.read_csv(COMPARACION_PATH)
    return df


@st.cache_data
def cargar_predicciones():
    df = pd.read_csv(PREDICCIONES_PATH)
    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"])
    return df


@st.cache_data
def cargar_residuos():
    df = pd.read_csv(RESIDUOS_PATH)
    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"])
    return df


def normalizar_columnas(df):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


df_comp = normalizar_columnas(cargar_comparacion())
df_pred = cargar_predicciones()
df_res = cargar_residuos()

st.subheader("1. Resultados comparativos")

st.write("""
La siguiente tabla resume el desempeño de los modelos evaluados mediante métricas como MAE,
RMSE, MAPE y R². Para este proyecto, el MAPE se considera una métrica importante porque permite
interpretar el error porcentual promedio de las predicciones.
""")

st.dataframe(df_comp, use_container_width=True)

st.divider()

# Detectar columnas principales de forma flexible
columnas = df_comp.columns.tolist()

col_modelo = None
col_etapa = None
col_mape = None
col_r2 = None

for col in columnas:
    col_lower = col.lower()

    if col_lower in ["modelo", "modelo_base", "model"]:
        col_modelo = col

    if col_lower in ["etapa", "tipo", "version", "versión"]:
        col_etapa = col

    if col_lower == "mape":
        col_mape = col

    if col_lower in ["r2", "r²", "r_2"]:
        col_r2 = col

if col_modelo is None:
    col_modelo = columnas[0]

if col_mape is None:
    st.error("No se encontró una columna llamada MAPE en comparacion_modelos.csv.")
    st.stop()

# Convertir MAPE a numérico
df_comp[col_mape] = pd.to_numeric(df_comp[col_mape], errors="coerce")

st.subheader("2. Ranking según MAPE")

df_ranking = df_comp.sort_values(by=col_mape, ascending=True).reset_index(drop=True)

st.dataframe(df_ranking, use_container_width=True)

mejor_modelo = df_ranking.iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Modelo ganador", str(mejor_modelo[col_modelo]))

with col2:
    st.metric("MAPE", f"{mejor_modelo[col_mape]:.2f}%")

with col3:
    if col_r2 is not None:
        st.metric("R²", f"{mejor_modelo[col_r2]:.3f}")
    else:
        st.metric("Criterio", "Menor MAPE")

st.success("""
El modelo con menor MAPE fue seleccionado como el modelo más adecuado para el despliegue,
ya que presentó el menor error porcentual promedio en el periodo de evaluación.
""")

st.divider()

st.subheader("3. Comparación visual del MAPE")

if col_etapa is not None:
    fig_mape = px.bar(
        df_comp,
        x=col_modelo,
        y=col_mape,
        color=col_etapa,
        barmode="group",
        title="Comparación del MAPE por modelo",
        labels={
            col_modelo: "Modelo",
            col_mape: "MAPE (%)",
            col_etapa: "Etapa"
        }
    )
else:
    fig_mape = px.bar(
        df_comp,
        x=col_modelo,
        y=col_mape,
        title="Comparación del MAPE por modelo",
        labels={
            col_modelo: "Modelo",
            col_mape: "MAPE (%)"
        }
    )

st.plotly_chart(fig_mape, use_container_width=True)

st.info("""
Un MAPE más bajo indica que el modelo tuvo un menor error porcentual promedio.
Por ello, esta métrica facilita comparar modelos aunque sus errores absolutos tengan escalas distintas.
""")

st.divider()

st.subheader("4. XGBoost: valores reales vs predichos")

st.write("""
El siguiente gráfico muestra el comportamiento de los valores reales frente a los valores predichos
por el modelo XGBoost seleccionado para el despliegue.
""")

df_grafico = df_pred.melt(
    id_vars="fecha_mes",
    value_vars=["real", "prediccion"],
    var_name="Serie",
    value_name="Visitantes"
)

fig_pred = px.line(
    df_grafico,
    x="fecha_mes",
    y="Visitantes",
    color="Serie",
    title="Visitantes reales vs visitantes predichos - XGBoost",
    labels={
        "fecha_mes": "Fecha",
        "Visitantes": "Total de visitantes",
        "Serie": "Serie"
    }
)

st.plotly_chart(fig_pred, use_container_width=True)

st.dataframe(df_pred, use_container_width=True)

st.divider()

st.subheader("5. Residuos del modelo XGBoost")

st.write("""
Los residuos representan la diferencia entre los valores reales y los valores predichos.
Su análisis permite observar en qué meses el modelo presentó mayor o menor desviación.
""")

fig_residuos = px.line(
    df_res,
    x="fecha_mes",
    y="residuo",
    title="Residuos del modelo XGBoost",
    labels={
        "fecha_mes": "Fecha",
        "residuo": "Residuo"
    }
)

st.plotly_chart(fig_residuos, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Residuo promedio", f"{df_res['residuo'].mean():,.2f}")

with col2:
    st.metric("Residuo mínimo", f"{df_res['residuo'].min():,.2f}")

with col3:
    st.metric("Residuo máximo", f"{df_res['residuo'].max():,.2f}")

st.divider()

st.subheader("Interpretación final")

st.write("""
De acuerdo con la comparación realizada, el modelo XGBoost fue seleccionado para el despliegue
en Streamlit porque obtuvo el mejor desempeño global según el MAPE. Además, su estructura
permite realizar predicciones en vivo a partir de un nuevo registro mensual, variables climáticas
agregadas y rezagos históricos.
""")