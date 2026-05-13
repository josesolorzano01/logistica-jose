import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq
 
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
 
RANDOM_STATE = 42
 
st.set_page_config(page_title="Riesgo actuarial", layout="centered")
st.title("Predicción de riesgo actuarial")
 
 
@st.cache_data
def cargar_base():
    csv = "insurance.csv" if os.path.exists("insurance.csv") else "insurance(2).csv"
    return pd.read_csv(csv)
 
 
@st.cache_resource
def entrenar_modelos(_df):
    """
    Reemplaza el .pkl: entrena KMeans para etiquetar y Regresión
    Logística para clasificar. Se ejecuta una sola vez por sesión.
    """
    numeric_features     = ["age", "bmi", "children", "charges"]
    categorical_features = ["sex", "smoker", "region"]
    all_features         = numeric_features + categorical_features
 
    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(),            numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ])
 
    # KMeans genera las etiquetas
    km_pipe = Pipeline([("pre", pre), ("km", KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10))])
    df2 = _df.copy()
    df2["cluster"] = km_pipe.fit_predict(df2[all_features])
 
    orden = df2.groupby("cluster")["charges"].mean().sort_values().index.tolist()
    mapa  = {orden[0]: "Bajo", orden[1]: "Medio", orden[2]: "Alto"}
    df2["riesgo_actuarial"] = df2["cluster"].map(mapa)
 
    # Regresión Logística aprende a clasificar con esas etiquetas
    pre2 = ColumnTransformer(transformers=[
        ("num", StandardScaler(),            numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ])
    lr_pipe = Pipeline([
        ("pre", pre2),
        ("lr",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    lr_pipe.fit(df2[all_features], df2["riesgo_actuarial"])
 
    metadata = {"nombre_modelo": "KMeans + Regresión Logística", "mapa_riesgo": mapa}
    return km_pipe, lr_pipe, df2, mapa, metadata
 
 
# ── Cargar datos y modelos ────────────────────────────────────────────────────
df           = cargar_base()
modelo_km, modelo_lr, df_model, mapa, metadata = entrenar_modelos(df)
 
st.caption(metadata["nombre_modelo"])
 
# ──────────────────────────────────────────────────────────────────────────────
# FORMULARIO (idéntico al original)
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
 
    # Predicción KMeans
    cluster   = int(modelo_km.predict(cliente)[0])
    riesgo_km = mapa.get(cluster, "No definido")
 
    # Predicción Regresión Logística
    riesgo_lr = modelo_lr.predict(cliente)[0]
    probas    = modelo_lr.predict_proba(cliente)[0]
    clases    = modelo_lr.classes_
 
    # ── Resultado (igual al original + métricas nuevas) ───────────────────
    icono = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}.get(riesgo_lr, "⚪")
    st.subheader(f"Riesgo actuarial: {icono} {riesgo_lr}")
 
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Regresión Logística", riesgo_lr)
    col_b.metric("KMeans",              f"{riesgo_km} (C{cluster})")
    col_c.metric("Probabilidad máx.",   f"{max(probas):.1%}")
 
    # Barra de probabilidades
    prob_df = pd.DataFrame({"Nivel": clases, "Probabilidad": probas}).sort_values("Probabilidad")
    fig_p, ax_p = plt.subplots(figsize=(6, 2.2))
    bar_colors = ["#2ecc71" if c=="Bajo" else "#f39c12" if c=="Medio" else "#e74c3c" for c in prob_df["Nivel"]]
    ax_p.barh(prob_df["Nivel"], prob_df["Probabilidad"], color=bar_colors)
    for i, v in enumerate(prob_df["Probabilidad"]):
        ax_p.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
    ax_p.set_xlim(0, 1.15)
    ax_p.set_xlabel("Probabilidad")
    ax_p.set_title("Probabilidad por nivel de riesgo")
    plt.tight_layout()
    st.pyplot(fig_p)
 
    # ── Recomendaciones con Groq (idéntico al original) ───────────────────
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
        - Nivel de riesgo actuarial: {riesgo_lr}
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
 
    # ── Visualización: comparativa + histograma ───────────────────────────
    st.divider()
    tab1, tab2 = st.tabs(["Comparativa del grupo", "Distribución de cargos"])
 
    with tab1:
        grupo   = df_model[df_model["riesgo_actuarial"] == riesgo_lr]
        resumen = grupo[["age","bmi","children","charges"]].agg(["mean","min","max"]).T.round(2)
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
        ax2.axvline(charges, color="black", linewidth=2, linestyle="--", label=f"Tu valor: ${charges:,.0f}")
        ax2.set_title("Distribución de cargos por nivel de riesgo")
        ax2.set_xlabel("Cargos ($)")
        ax2.set_ylabel("Frecuencia")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
 
# ── Tabla original (siempre visible) ─────────────────────────────────────────
st.divider()
st.write("Vista rápida de la base principal")
st.dataframe(df.head(20), use_container_width=True)
