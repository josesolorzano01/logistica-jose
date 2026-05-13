import os
import pandas as pd
import numpy as np
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

st.set_page_config(page_title="Riesgo Actuarial", layout="centered")
st.title("🛡️ Predicción de Riesgo Actuarial")
st.caption("Modelo: Regresión Logística | Dataset: insurance.csv")

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_base():
    csv = "insurance.csv" if os.path.exists("insurance.csv") else "insurance(2).csv"
    return pd.read_csv(csv)

# ── Entrenamiento del modelo (KMeans etiqueta → Regresión Logística clasifica) ─
@st.cache_resource
def entrenar_modelo(df):
    numeric_features     = ["age", "bmi", "children", "charges"]
    categorical_features = ["sex", "smoker", "region"]
    all_features         = numeric_features + categorical_features

    preprocessing = ColumnTransformer(transformers=[
        ("num", StandardScaler(),            numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ])

    # KMeans genera las etiquetas Bajo / Medio / Alto
    kmeans_pipe = Pipeline([
        ("pre", preprocessing),
        ("km",  KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)),
    ])
    df = df.copy()
    df["cluster"] = kmeans_pipe.fit_predict(df[all_features])

    orden = df.groupby("cluster")["charges"].mean().sort_values().index.tolist()
    mapa  = {orden[0]: "Bajo", orden[1]: "Medio", orden[2]: "Alto"}
    df["riesgo_actuarial"] = df["cluster"].map(mapa)

    # Regresión Logística aprende a clasificar
    preprocessing2 = ColumnTransformer(transformers=[
        ("num", StandardScaler(),            numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ])
    lr_pipe = Pipeline([
        ("pre", preprocessing2),
        ("lr",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    lr_pipe.fit(df[all_features], df["riesgo_actuarial"])

    return lr_pipe, df, mapa

df = cargar_base()
modelo, df_model, mapa = entrenar_modelo(df)

# ──────────────────────────────────────────────────────────────────────────────
# 1. FORMULARIO DE INGRESO DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
st.header("📋 Ingreso de datos del cliente")

with st.form("datos"):
    col1, col2 = st.columns(2)
    age      = col1.number_input("Edad",                    18,    100,   35)
    sex      = col2.selectbox("Sexo",                       sorted(df["sex"].unique()))
    bmi      = col1.number_input("IMC (BMI)",               10.0,  60.0,  28.0, step=0.1)
    children = col2.number_input("Número de hijos",         0,     10,    1)
    smoker   = col1.selectbox("¿Fumador?",                  sorted(df["smoker"].unique()))
    region   = col2.selectbox("Región",                     sorted(df["region"].unique()))
    charges  = st.number_input("Cargos médicos estimados ($)", 0.0, 100_000.0, 12_000.0, step=500.0)
    enviar   = st.form_submit_button("🔍 Evaluar riesgo", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. PREDICCIÓN DEL RIESGO ACTUARIAL
# ──────────────────────────────────────────────────────────────────────────────
if enviar:
    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker,
        "region": region, "charges": charges,
    }])

    riesgo = modelo.predict(cliente)[0]
    probas = modelo.predict_proba(cliente)[0]
    clases = modelo.classes_

    icono = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}.get(riesgo, "⚪")
    st.header(f"Resultado: {icono} Riesgo **{riesgo}**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Nivel de riesgo", riesgo)
    col_b.metric("Probabilidad",    f"{max(probas):.1%}")
    col_c.metric("IMC ingresado",   f"{bmi:.1f}")

    # Barra de probabilidades
    prob_df = pd.DataFrame({"Nivel": clases, "Probabilidad": probas}).sort_values("Probabilidad", ascending=True)
    fig_p, ax_p = plt.subplots(figsize=(6, 2.2))
    colores_barra = ["#2ecc71" if c == "Bajo" else "#f39c12" if c == "Medio" else "#e74c3c" for c in prob_df["Nivel"]]
    ax_p.barh(prob_df["Nivel"], prob_df["Probabilidad"], color=colores_barra)
    for i, v in enumerate(prob_df["Probabilidad"]):
        ax_p.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
    ax_p.set_xlim(0, 1.15)
    ax_p.set_xlabel("Probabilidad")
    ax_p.set_title("Probabilidad por nivel de riesgo")
    plt.tight_layout()
    st.pyplot(fig_p)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RECOMENDACIONES GENERADAS CON API (Groq)
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("💡 Recomendaciones actuariales")

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    if api_key:
        prompt = f"""
Actúa como analista actuarial senior. Explica el resultado y da 3 recomendaciones
claras y profesionales para el siguiente perfil:

- Edad: {age} años | Sexo: {sex} | IMC: {bmi}
- Hijos: {children} | Fumador: {smoker} | Región: {region}
- Cargos médicos estimados: ${charges:,.0f}
- Nivel de riesgo actuarial: {riesgo}

Responde en español, de forma concreta y profesional.
"""
        try:
            client     = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Eres un analista actuarial prudente, claro y profesional."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.4,
                max_tokens=600,
            )
            st.info(completion.choices[0].message.content)
        except Exception as e:
            st.warning(f"No se pudo generar recomendación con Groq: {e}")
    else:
        rec = {
            "Bajo":  "✅ Prima competitiva recomendada.\n\n✅ Cobertura completa sin exclusiones.\n\n✅ Programa de fidelización por renovación.\n\n✅ Revisión anual preventiva.",
            "Medio": "⚠️ Prima estándar con recargo del 10–15%.\n\n⚠️ Inscribir en programa preventivo de salud.\n\n⚠️ Seguimiento semestral de indicadores.\n\n⚠️ Descuento si mejora factores de riesgo en 6 meses.",
            "Alto":  "🔴 Recargo actuarial del 30–50% sobre tarifa base.\n\n🔴 Examen médico previo obligatorio.\n\n🔴 Cláusulas de coaseguro en siniestros de alto costo.\n\n🔴 Plan de cesación tabáquica como condición contractual.",
        }
        st.info(rec.get(riesgo, ""))
        st.caption("Agrega `GROQ_API_KEY` en los secretos de Streamlit para recomendaciones con IA.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. VISUALIZACIÓN DE DATOS
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Visualización")

    tab1, tab2, tab3 = st.tabs(["Comparativa del grupo", "Distribución de cargos", "Tabla del dataset"])

    with tab1:
        grupo_filtrado = df_model[df_model["riesgo_actuarial"] == riesgo]
        resumen = grupo_filtrado[["age", "bmi", "children", "charges"]].agg(["mean", "min", "max"]).T.round(2)
        resumen.columns = ["Promedio del grupo", "Mínimo", "Máximo"]
        resumen["Tu valor"] = [age, bmi, children, charges]
        st.write(f"**Tú vs. grupo {riesgo}** ({len(grupo_filtrado):,} clientes)")
        st.dataframe(resumen, use_container_width=True)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colores = {"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"}
        for nivel, color in colores.items():
            datos = df_model[df_model["riesgo_actuarial"] == nivel]["charges"]
            ax2.hist(datos, bins=35, alpha=0.6, color=color, label=nivel, edgecolor="white")
        ax2.axvline(charges, color="black", linewidth=2, linestyle="--", label=f"Tu valor: ${charges:,.0f}")
        ax2.set_title("Distribución de cargos médicos por nivel de riesgo")
        ax2.set_xlabel("Cargos ($)")
        ax2.set_ylabel("Frecuencia")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        st.write("Primeros 30 registros del dataset")
        st.dataframe(df.head(30), use_container_width=True)

else:
    st.divider()
    st.subheader("📂 Vista rápida del dataset")
    st.dataframe(df.head(20), use_container_width=True)
