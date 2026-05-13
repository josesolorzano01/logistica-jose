import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42

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


# ── Regresión Logística entrenada sobre las etiquetas del KMeans ──────────────
@st.cache_resource
def entrenar_lr(df, mapa_kmeans):
    """Usa las predicciones del KMeans como etiquetas para entrenar
    una Regresión Logística supervisada."""
    numeric_features     = ["age", "bmi", "children", "charges"]
    categorical_features = ["sex", "smoker", "region"]
    all_features         = numeric_features + categorical_features

    modelo_km, _ = cargar_modelo()

    df = df.copy()
    df["cluster"] = modelo_km.predict(df[all_features])
    df["riesgo_actuarial"] = df["cluster"].map(mapa_kmeans)

    preprocessing = ColumnTransformer(transformers=[
        ("num", StandardScaler(),            numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ])

    lr_pipe = Pipeline([
        ("pre", preprocessing),
        ("lr",  LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])
    lr_pipe.fit(df[all_features], df["riesgo_actuarial"])
    return lr_pipe, df


# ── Cargar todo ───────────────────────────────────────────────────────────────
modelo_km, metadata = cargar_modelo()
df = cargar_base()
mapa = {int(k): v for k, v in metadata["mapa_riesgo"].items()}
modelo_lr, df_model = entrenar_lr(df, mapa)

st.caption(metadata["nombre_modelo"] + " · Regresión Logística")

# ──────────────────────────────────────────────────────────────────────────────
# FORMULARIO (igual que el original)
# ──────────────────────────────────────────────────────────────────────────────
with st.form("datos"):
    col1, col2 = st.columns(2)

    age      = col1.number_input("Edad",   18,   100,   35)
    sex      = col2.selectbox("Sexo",      sorted(df["sex"].unique()))
    bmi      = col1.number_input("BMI",    10.0, 60.0,  28.0)
    children = col2.number_input("Hijos",  0,    10,    1)
    smoker   = col1.selectbox("Fumador",   sorted(df["smoker"].unique()))
    region   = col2.selectbox("Región",    sorted(df["region"].unique()))
    charges  = st.number_input("Cargos médicos estimados", 0.0, 100000.0, 12000.0)

    enviar = st.form_submit_button("Evaluar")


if enviar:
    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker,
        "region": region, "charges": charges,
    }])

    # ── Predicción KMeans (original) ──────────────────────────────────────
    cluster = int(modelo_km.predict(cliente)[0])
    riesgo_km = mapa.get(cluster, "No definido")

    # ── Predicción Regresión Logística (nueva) ────────────────────────────
    riesgo_lr = modelo_lr.predict(cliente)[0]
    probas    = modelo_lr.predict_proba(cliente)[0]
    clases    = modelo_lr.classes_

    # ── Resultado principal ───────────────────────────────────────────────
    icono = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}.get(riesgo_lr, "⚪")

    st.subheader(f"Riesgo actuarial: {icono} {riesgo_lr}")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Regresión Logística", riesgo_lr)
    col_b.metric("KMeans (cluster)",    f"{riesgo_km} (C{cluster})")
    col_c.metric("Probabilidad máx.",   f"{max(probas):.1%}")

    # Barra de probabilidades por clase
    prob_df = (
        pd.DataFrame({"Nivel": clases, "Probabilidad": probas})
        .sort_values("Probabilidad", ascending=True)
    )
    fig_p, ax_p = plt.subplots(figsize=(6, 2.2))
    bar_colors = [
        "#2ecc71" if c == "Bajo" else "#f39c12" if c == "Medio" else "#e74c3c"
        for c in prob_df["Nivel"]
    ]
    ax_p.barh(prob_df["Nivel"], prob_df["Probabilidad"], color=bar_colors)
    for i, v in enumerate(prob_df["Probabilidad"]):
        ax_p.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
    ax_p.set_xlim(0, 1.15)
    ax_p.set_xlabel("Probabilidad")
    ax_p.set_title("Probabilidad por nivel de riesgo (Regresión Logística)")
    plt.tight_layout()
    st.pyplot(fig_p)

    # ──────────────────────────────────────────────────────────────────────
    # RECOMENDACIONES CON API (igual que el original, prompt mejorado)
    # ──────────────────────────────────────────────────────────────────────
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
        - Cluster asignado (KMeans): {cluster}
        - Nivel de riesgo actuarial (Regresión Logística): {riesgo_lr}
        """
        try:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Eres un analista actuarial prudente, claro y profesional."},
                    {"role": "user",   "content": prompt},
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

    # ──────────────────────────────────────────────────────────────────────
    # VISUALIZACIÓN: comparativa del grupo + histograma
    # ──────────────────────────────────────────────────────────────────────
    st.divider()
    tab1, tab2 = st.tabs(["Comparativa del grupo", "Distribución de cargos"])

    with tab1:
        grupo = df_model[df_model["riesgo_actuarial"] == riesgo_lr]
        resumen = grupo[["age", "bmi", "children", "charges"]].agg(["mean", "min", "max"]).T.round(2)
        resumen.columns = ["Promedio del grupo", "Mínimo", "Máximo"]
        resumen["Tu valor"] = [age, bmi, children, charges]
        st.write(f"**Tú vs. grupo {riesgo_lr}** ({len(grupo):,} clientes similares)")
        st.dataframe(resumen, use_container_width=True)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colores = {"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"}
        for nivel, color in colores.items():
            datos = df_model[df_model["riesgo_actuarial"] == nivel]["charges"]
            ax2.hist(datos, bins=35, alpha=0.6, color=color, label=nivel, edgecolor="white")
        ax2.axvline(charges, color="black", linewidth=2, linestyle="--",
                    label=f"Tu valor: ${charges:,.0f}")
        ax2.set_title("Distribución de cargos por nivel de riesgo")
        ax2.set_xlabel("Cargos ($)")
        ax2.set_ylabel("Frecuencia")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

# ──────────────────────────────────────────────────────────────────────────────
# TABLA ORIGINAL (siempre visible)
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.write("Vista rápida de la base principal")
st.dataframe(df.head(20), use_container_width=True)
