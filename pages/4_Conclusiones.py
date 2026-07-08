import streamlit as st

st.title("Conclusiones del proyecto")

st.write("""
En esta sección se resumen los principales aprendizajes del proyecto, considerando tanto
el valor para la gestión de museos como la parte técnica del modelo predictivo desplegado.
""")

st.divider()


st.write("""1. En conclusión, la solución permite que la asistencia a museos no se analice solo como un dato
histórico, sino como una información útil para anticipar posibles niveles de visitantes en los
siguientes meses. Esto puede apoyar la planificación de personal, horarios, campañas, recursos
y actividades culturales, especialmente en meses donde se espera mayor o menor demanda.
""")


st.write("""
2. En síntesis, el modelo XGBoost fue elegido para el despliegue porque tuvo el mejor desempeño
global en la comparación de modelos y, además, es más práctico para realizar predicciones en vivo.
La aplicación permite ingresar un nuevo registro mensual, actualizar las variables de historial
y estimar la asistencia del siguiente mes sin tener que volver a entrenar el modelo en ese momento.
""")


st.write("""
3. Se resume que Streamlit permitió transformar el trabajo realizado en Colab en una aplicación
más entendible y fácil de usar. El notebook queda como evidencia del proceso técnico, mientras
que la app funciona como una versión más cercana a un entorno productivo, donde el usuario puede
visualizar datos, comparar modelos y probar una predicción de manera directa.
""")

st.divider()

st.subheader("Limitaciones")

st.write("""
La predicción se realiza a nivel mensual agregado, por lo que no representa todavía una predicción
individual por museo. Además, las variables climáticas se manejan como valores mensuales agregados,
no como el clima exacto de cada ubicación. Para una versión más avanzada, se podría trabajar una
predicción por museo, departamento o zona geográfica.
""")

st.subheader("Mejoras futuras")

st.write("""
Como mejora futura, se podría agregar almacenamiento permanente para los nuevos registros,
actualización automática de datos, predicción por museo y reentrenamiento programado del modelo.
También se podría mejorar el manejo de variables externas, como feriados, eventos culturales o
temporadas turísticas.
""")