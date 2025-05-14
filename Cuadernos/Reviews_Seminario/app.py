import streamlit as st
import pandas as pd
import pickle
import re
import unicodedata

# ---------- CARGAR MODELO Y VECTORIZADOR ----------
try:
    with open("vectorizador.pkl", "rb") as f:
        vectorizador = pickle.load(f)

    with open("modelo_naive_bayes.pkl", "rb") as f:
        modelo = pickle.load(f)
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos 'vectorizador.pkl' o 'modelo_naive_bayes.pkl'. Asegúrate de que estén en la misma carpeta que este archivo app.py.")
    st.stop()

# ---------- FUNCIONES ----------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"\W", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def predecir_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)
    vector = vectorizador.transform([texto_limpio])
    pred = modelo.predict(vector)[0]
    return pred

# ---------- INTERFAZ ----------
st.set_page_config(page_title="Análisis de Reseñas", layout="wide")
st.title("📊 Dashboard de Sentimiento de Reseñas")
st.markdown("Este sistema analiza automáticamente reseñas cargadas desde un archivo `.csv` y detecta si son **positivas** o **negativas**.")

# ---------- CARGA DE ARCHIVO ----------
archivo = st.file_uploader("📂 Sube un archivo .csv con una columna de reseñas", type=["csv"])

if archivo is not None:
    try:
        df = pd.read_csv(archivo)

        # Selección de columna
        columnas = df.columns.tolist()
        col_resena = st.selectbox("Selecciona la columna que contiene las reseñas:", columnas)

        if st.button("🔍 Analizar reseñas"):
            df["reseña_limpia"] = df[col_resena].astype(str).apply(limpiar_texto)
            vectores = vectorizador.transform(df["reseña_limpia"])
            predicciones = modelo.predict(vectores)

            df["Predicción"] = predicciones
            df["Sentimiento"] = df["Predicción"].apply(lambda x: "✅ Positiva" if x == 1 else "⚠️ Negativa")

            st.success("✅ Análisis completado")
            st.dataframe(df[[col_resena, "Sentimiento"]], use_container_width=True)

            # Métricas
            total = len(df)
            positivas = sum(df["Predicción"] == 1)
            negativas = sum(df["Predicción"] == 0)

            st.markdown("### 📈 Resumen del Análisis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Reseñas", total)
            col2.metric("Positivas", positivas)
            col3.metric("Negativas", negativas)

            # Descarga
            csv_resultado = df[[col_resena, "Sentimiento"]].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar resultados como CSV", data=csv_resultado, file_name="resultados_reseñas.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
