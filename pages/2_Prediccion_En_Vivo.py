import streamlit as st
import pandas as pd
import json
import xgboost as xgb
from pathlib import Path

st.title("Predicción en vivo de asistencia mensual")

st.write("""
Esta vista permite agregar un nuevo registro mensual al historial de la sesión y predecir
la asistencia del siguiente mes mediante el modelo XGBoost seleccionado para el despliegue.
""")

DATA_PATH = Path("data/dataset_mensual_museos.csv")
MODEL_PATH = Path("models/modelo_xgb_con_api.json")
METADATA_PATH = Path("models/metadata_modelo.json")


@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"])
    df = df.sort_values("fecha_mes").reset_index(drop=True)
    return df


@st.cache_resource
def cargar_modelo():
    modelo = xgb.XGBRegressor()
    modelo.load_model(MODEL_PATH)
    return modelo


@st.cache_data
def cargar_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return "Verano"
    elif mes in [3, 4, 5]:
        return "Otoño"
    elif mes in [6, 7, 8]:
        return "Invierno"
    else:
        return "Primavera"


def preparar_historial(df):
    columnas_necesarias = [
        "fecha_mes",
        "TOTAL",
        "ANIO",
        "COD_MES",
        "ESTACION",
        "TEMP_PROMEDIO_MES",
        "PRECIPITACION_MES"
    ]

    df_hist = df[columnas_necesarias].copy()
    df_hist["fecha_mes"] = pd.to_datetime(df_hist["fecha_mes"])
    df_hist = df_hist.sort_values("fecha_mes").reset_index(drop=True)

    return df_hist


def construir_registro_base(df_hist, fecha_pred, temp_pred, precip_pred):
    fecha_pred = pd.to_datetime(fecha_pred)

    fecha_lag_1 = fecha_pred - pd.DateOffset(months=1)
    fecha_lag_12 = fecha_pred - pd.DateOffset(months=12)

    fila_lag_1 = df_hist[df_hist["fecha_mes"] == fecha_lag_1]
    fila_lag_12 = df_hist[df_hist["fecha_mes"] == fecha_lag_12]

    if fila_lag_1.empty:
        raise ValueError(f"No existe información del mes anterior: {fecha_lag_1.strftime('%Y-%m')}")

    if fila_lag_12.empty:
        raise ValueError(f"No existe información del mismo mes del año anterior: {fecha_lag_12.strftime('%Y-%m')}")

    ultimos_3 = df_hist[df_hist["fecha_mes"] < fecha_pred].tail(3)

    if len(ultimos_3) < 3:
        raise ValueError("No existen suficientes meses históricos para calcular el promedio móvil de 3 meses.")

    registro = pd.DataFrame([{
        "ANIO": fecha_pred.year,
        "COD_MES": fecha_pred.month,
        "ESTACION": obtener_estacion(fecha_pred.month),
        "TEMP_PROMEDIO_MES": temp_pred,
        "PRECIPITACION_MES": precip_pred,
        "TOTAL_LAG_1": fila_lag_1["TOTAL"].iloc[0],
        "TOTAL_LAG_12": fila_lag_12["TOTAL"].iloc[0],
        "TOTAL_ROLLING_3": ultimos_3["TOTAL"].mean()
    }])

    return registro


def transformar_para_modelo(registro, metadata):
    # Compatibilidad con dos posibles formatos de metadata
    columnas_numericas = metadata.get("columnas_numericas", metadata.get("numericas"))
    columnas_categoricas = metadata.get("columnas_categoricas", metadata.get("categoricas"))

    if columnas_numericas is None:
        columnas_numericas = [
            "ANIO",
            "COD_MES",
            "TEMP_PROMEDIO_MES",
            "PRECIPITACION_MES",
            "TOTAL_LAG_1",
            "TOTAL_LAG_12",
            "TOTAL_ROLLING_3"
        ]

    if columnas_categoricas is None:
        columnas_categoricas = ["ESTACION"]

    partes = []

    # Variables numéricas
    partes.append(registro[columnas_numericas].reset_index(drop=True))

    # Variable categórica ESTACION
    # En el notebook se usó OneHotEncoder(drop="first"), por eso se elimina la primera categoría.
    for col in columnas_categoricas:
        if "categorias_por_columna" in metadata:
            categorias = metadata["categorias_por_columna"][col]
            drop_idx = metadata.get("drop_idx_por_columna", {}).get(col, 0)
        else:
            categorias = metadata.get(
                "categorias_estacion",
                ["Invierno", "Otoño", "Primavera", "Verano"]
            )
            drop_idx = 0

        for i, categoria in enumerate(categorias):
            if drop_idx is not None and i == drop_idx:
                continue

            nombre_columna = f"{col}_{categoria}"
            valor = int(registro[col].iloc[0] == categoria)
            partes.append(pd.DataFrame({nombre_columna: [valor]}))

    registro_transformado = pd.concat(partes, axis=1)

    return registro_transformado

df_base = cargar_datos()
modelo = cargar_modelo()
metadata = cargar_metadata()

if "historial_museos" not in st.session_state:
    st.session_state.historial_museos = preparar_historial(df_base)

df_hist = st.session_state.historial_museos.copy()

st.subheader("1. Historial actual")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Primer mes", df_hist["fecha_mes"].min().strftime("%Y-%m"))

with col2:
    st.metric("Último mes registrado", df_hist["fecha_mes"].max().strftime("%Y-%m"))

with col3:
    st.metric("Meses disponibles", len(df_hist))

st.write("Últimos 12 meses registrados:")

st.dataframe(df_hist.tail(12), use_container_width=True)

st.divider()

st.subheader("2. Agregar nuevo registro mensual")

st.write("""
Este paso simula la llegada de un nuevo dato real al sistema. El registro se agrega al historial
de la sesión y servirá para calcular la predicción del siguiente mes.
""")

ultimo_mes = df_hist["fecha_mes"].max()
siguiente_registro = ultimo_mes + pd.DateOffset(months=1)

with st.form("form_nuevo_registro"):
    col1, col2 = st.columns(2)

    with col1:
        anio_reg = st.number_input(
            "Año del nuevo registro",
            min_value=2007,
            max_value=2035,
            value=int(siguiente_registro.year)
        )

        mes_reg = st.selectbox(
            "Mes del nuevo registro",
            list(range(1, 13)),
            index=int(siguiente_registro.month) - 1
        )

        total_reg = st.number_input(
            "Total real de visitantes del mes",
            min_value=0,
            value=250000
        )

    with col2:
        temp_reg = st.number_input(
            "Temperatura promedio del mes",
            value=18.5
        )

        precip_reg = st.number_input(
            "Precipitación mensual",
            value=2.0
        )

    agregar = st.form_submit_button("Agregar registro al historial")

if agregar:
    fecha_reg = pd.Timestamp(
        year=int(anio_reg),
        month=int(mes_reg),
        day=1
    )

    nuevo_registro = pd.DataFrame([{
        "fecha_mes": fecha_reg,
        "TOTAL": total_reg,
        "ANIO": int(anio_reg),
        "COD_MES": int(mes_reg),
        "ESTACION": obtener_estacion(int(mes_reg)),
        "TEMP_PROMEDIO_MES": temp_reg,
        "PRECIPITACION_MES": precip_reg
    }])

    df_actualizado = pd.concat(
        [st.session_state.historial_museos, nuevo_registro],
        ignore_index=True
    )

    df_actualizado = (
        df_actualizado
        .sort_values("fecha_mes")
        .drop_duplicates(subset=["fecha_mes"], keep="last")
        .reset_index(drop=True)
    )

    st.session_state.historial_museos = df_actualizado

    st.success(f"Registro agregado correctamente para {fecha_reg.strftime('%Y-%m')}.")

st.divider()

st.subheader("3. Predecir asistencia del siguiente mes")

df_hist_actual = st.session_state.historial_museos.copy()
ultimo_mes_actual = df_hist_actual["fecha_mes"].max()
mes_a_predecir = ultimo_mes_actual + pd.DateOffset(months=1)

st.write(f"Con el historial actual, el sistema predecirá el mes: **{mes_a_predecir.strftime('%Y-%m')}**")

with st.form("form_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        temp_pred = st.number_input(
            "Temperatura promedio esperada",
            value=18.0
        )

    with col2:
        precip_pred = st.number_input(
            "Precipitación mensual esperada",
            value=2.5
        )

    predecir = st.form_submit_button("Predecir asistencia")

if predecir:
    try:
        registro_base = construir_registro_base(
            df_hist=df_hist_actual,
            fecha_pred=mes_a_predecir,
            temp_pred=temp_pred,
            precip_pred=precip_pred
        )

        registro_modelo = transformar_para_modelo(
            registro=registro_base,
            metadata=metadata
        )

        prediccion = modelo.predict(registro_modelo)[0]
        prediccion = max(0, prediccion)

        st.success(f"Predicción realizada para {mes_a_predecir.strftime('%Y-%m')}")

        st.metric(
            label="Visitantes estimados",
            value=f"{prediccion:,.0f}"
        )

        st.subheader("Variables interpretables utilizadas")
        st.dataframe(registro_base, use_container_width=True)

        st.subheader("Variables transformadas para el modelo")
        st.dataframe(registro_modelo, use_container_width=True)

    except ValueError as error:
        st.error(str(error))

st.divider()

st.subheader("4. Reiniciar historial")

if st.button("Restaurar historial original"):
    st.session_state.historial_museos = preparar_historial(df_base)
    st.success("Historial restaurado correctamente.")