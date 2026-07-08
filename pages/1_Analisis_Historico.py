import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Análisis histórico de asistencia a museos")

st.write("""
Esta vista permite analizar la evolución mensual del total de visitantes registrados en los museos.
Los datos provienen del dataset procesado previamente en Google Colab.
""")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("data/dataset_mensual_museos.csv")
    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"])
    return df

df = cargar_datos()

# Métricas principales
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Primer mes registrado",
        df["fecha_mes"].min().strftime("%Y-%m")
    )

with col2:
    st.metric(
        "Último mes registrado",
        df["fecha_mes"].max().strftime("%Y-%m")
    )

with col3:
    st.metric(
        "Total histórico de visitantes",
        f"{df['TOTAL'].sum():,.0f}"
    )

st.divider()

# Filtro por año
st.subheader("Filtro de periodo")

anio_min = int(df["ANIO"].min())
anio_max = int(df["ANIO"].max())

rango_anios = st.slider(
    "Selecciona el rango de años",
    min_value=anio_min,
    max_value=anio_max,
    value=(anio_min, anio_max)
)

df_filtrado = df[
    (df["ANIO"] >= rango_anios[0]) &
    (df["ANIO"] <= rango_anios[1])
]

# Gráfico de evolución mensual
st.subheader("Evolución mensual de visitantes")

fig_linea = px.line(
    df_filtrado,
    x="fecha_mes",
    y="TOTAL",
    title="Evolución mensual del total de visitantes",
    labels={
        "fecha_mes": "Fecha",
        "TOTAL": "Total de visitantes"
    }
)

st.plotly_chart(fig_linea, use_container_width=True)

# Visitantes acumulados por año
st.subheader("Visitantes acumulados por año")

df_anual = (
    df_filtrado
    .groupby("ANIO", as_index=False)["TOTAL"]
    .sum()
)

fig_anual = px.bar(
    df_anual,
    x="ANIO",
    y="TOTAL",
    title="Total de visitantes por año",
    labels={
        "ANIO": "Año",
        "TOTAL": "Total de visitantes"
    }
)

st.plotly_chart(fig_anual, use_container_width=True)

# Visitantes por mes
st.subheader("Comportamiento acumulado por mes")

df_mes = (
    df_filtrado
    .groupby("COD_MES", as_index=False)["TOTAL"]
    .sum()
)

fig_mes = px.bar(
    df_mes,
    x="COD_MES",
    y="TOTAL",
    title="Visitantes acumulados por mes",
    labels={
        "COD_MES": "Mes",
        "TOTAL": "Total de visitantes"
    }
)

st.plotly_chart(fig_mes, use_container_width=True)

# Tabla
st.subheader("Datos históricos filtrados")

st.dataframe(
    df_filtrado,
    use_container_width=True
)