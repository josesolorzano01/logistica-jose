import os
import json
import joblib
import pandas as pd
import streamlit as st
from groq import Groq
 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
 
 
st.set_page_config(page_title="Riesgo actuarial", layout="centered")
st.title("Predicción de riesgo actuarial")
 
 
@st.cache_resource
def cargar_modelo():
    pkl = (
        "kmeans_riesgo_actuarial.pkl"
        if os.path.exists("kmeans_riesgo_actuarial.pkl")
        else "kmeans_riesgo_actuarial(2).pkl"
    )
 
    meta = (
        "model_metadata.json"
        if os.path.exists("model_metadata.json")
        else "model_metadata(2).json"
    )
 
    modelo = joblib.load(pkl)
 
    with open(meta, encoding="utf-8") as f:
        metadata = json.load(f)
 
    return modelo, metadata
 
 
@st.cache_data
def cargar_base():
    csv = "insurance.csv" if os.path.exists("insurance.csv") else "insurance(2).csv"
    return pd.read_csv(csv)
 
 
# ── NUEVO: entrena Regresión Logística usando etiquetas del KMeans ────────────
@st.cache_resource
def entrenar_lr(_modelo_km, _df, _mapa):
    numeric_features     = ["age", "bmi", "children", "charges"]
    categorical_features = ["sex", "smoker", "region"]
    all_features         = numeric_features + categorical_features
 
    df2 = _df.copy()
    df2["cluster"] = _modelo_km.predict(df2[all_features])
    df2["riesgo"]  = df2["cluster"].map(_mapa)
 
    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(),            numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ])
    lr = Pipeline([("pre", pre),
                   ("lr",  LogisticRegression(max_iter=1000, class_weight="balanced"))])
    lr.fit(df2[all_features], df2["riesgo"])
    return lr
# ─────────────────────────────────────────────────────────────────────────────
 
 
modelo, metadata = cargar_modelo()
df = cargar_base()
mapa = {int(k): v for k, v in metadata["mapa_riesgo"].items()}
modelo_lr = entrenar_lr(modelo, df, mapa)   # ← NUEVO
 
st.caption(metadata["nombre_modelo"])
 
with st.form("datos"):
    col1, col2 = st.columns(2)
 
    age = col1.number_input("Edad", 18, 100, 35)
    sex = col2.selectbox("Sexo", sorted(df["sex"].unique()))
    bmi = col1.number_input("BMI", 10.0, 60.0, 28.0)
    children = col2.number_input("Hijos", 0, 10, 1)
    smoker = col1.selectbox("Fumador", sorted(df["smoker"].unique()))
    region = col2.selectbox("Región", sorted(df["region"].unique()))
    charges = st.number_input(
        "Cargos médicos estimados",
        0.0,
        100000.0,
        12000.0
    )
 
    enviar = st.form_submit_button("Evaluar")
 
 
if enviar:
    cliente = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges,
        }
    ])
 
    cluster = int(modelo.predict(cliente)[0])
    riesgo = mapa.get(cluster, "No definido")
 
    st.subheader(f"Riesgo actuarial: {riesgo}")
    st.write(f"Cluster asignado: {cluster}")
 
    # ── NUEVO: clasificación con Regresión Logística ──────────────────────
    riesgo_lr = modelo_lr.predict(cliente)[0]
    probas    = modelo_lr.predict_proba(cliente)[0]
    clases    = modelo_lr.classes_
 
    icono = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}.get(riesgo_lr, "⚪")
    st.subheader(f"Regresión Logística: {icono} {riesgo_lr}")
 
    prob_tabla = pd.DataFrame({
        "Nivel de riesgo": clases,
        "Probabilidad":    [f"{p:.1%}" for p in probas],
    }).sort_values("Nivel de riesgo").reset_index(drop=True)
    st.table(prob_tabla)
    # ─────────────────────────────────────────────────────────────────────
 
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
 
    if api_key:
        prompt = f"""
        Actúa como analista actuarial.
 
        Explica brevemente el resultado del modelo y brinda 3 recomendaciones prudentes,
        claras y profesionales para el usuario.
 
        Datos del cliente:
        - Edad: {age}
        - Sexo: {sex}
        - BMI: {bmi}
        - Hijos: {children}
        - Fumador: {smoker}
        - Región: {region}
        - Cargos médicos estimados: {charges}
 
        Resultado del modelo:
        - Cluster asignado: {cluster}
        - Nivel de riesgo actuarial: {riesgo}
        """
 
        try:
            client = Groq(api_key=api_key)
 
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista actuarial prudente, claro y profesional.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.4,
                max_tokens=500,
            )
 
            respuesta = completion.choices[0].message.content
            st.info(respuesta)
 
        except Exception as e:
            st.warning(f"No se pudo generar recomendación con Groq: {e}")
 
    else:
        st.warning("Agregue GROQ_API_KEY en los secretos de Streamlit.")
 
 
st.divider()
st.write("Vista rápida de la base principal")
st.dataframe(df.head(20), use_container_width=True)
