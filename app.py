import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(page_title="Riesgo Actuarial", layout="centered")
st.title("🛡️ Predicción de Riesgo Actuarial")
st.caption("Modelo K-means | Dataset: insurance.csv")

# ── Carga de modelo y datos ───────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    pkl = (
        "kmeans_riesgo_actuarial.pkl"
        if os.path.exists("kmeans_riesgo_actuarial.pkl")
        else "kmeans_riesgo_actuarial.pkl"
    )
    meta = (
        "model_metadata.json"
        if os.path.exists("model_metadata.json")
        else "model_metadata.json"
    )
    modelo = joblib.load(pkl)
    with open(meta, encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata

@st.cache_data
def cargar_base():
    csv = "insurance.csv" if os.path.exists("insurance.csv") else "insurance.csv"
    return pd.read_csv(csv)

modelo, metadata = cargar_modelo()
df = cargar_base()
mapa = {int(k): v for k, v in metadata["mapa_riesgo"].items()}

# ──────────────────────────────────────────────────────────────────────────────
# 1. FORMULARIO DE INGRESO DE DATOS
# ──────────────────────────────────────────────────────────────────────────────
st.header("📋 Ingreso de datos del cliente")

with st.form("datos"):
    col1, col2 = st.columns(2)

    age      = col1.number_input("Edad",                18,    100,   35)
    sex      = col2.selectbox("Sexo",                   sorted(df["sex"].unique()))
    bmi      = col1.number_input("IMC (BMI)",           10.0,  60.0,  28.0, step=0.1)
    children = col2.number_input("Número de hijos",     0,     10,    1)
    smoker   = col1.selectbox("¿Fumador?",              sorted(df["smoker"].unique()))
    region   = col2.selectbox("Región",                 sorted(df["region"].unique()))
    charges  = st.number_input("Cargos médicos estimados ($)", 0.0, 100_000.0, 12_000.0, step=500.0)

    enviar = st.form_submit_button("🔍 Evaluar riesgo", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2. PREDICCIÓN DEL RIESGO ACTUARIAL
# ──────────────────────────────────────────────────────────────────────────────
if enviar:
    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker,
        "region": region, "charges": charges,
    }])

    cluster = int(modelo.predict(cliente)[0])
    riesgo  = mapa.get(cluster, "No definido")

    icono = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}.get(riesgo, "⚪")

    st.header(f"Resultado: {icono} Riesgo **{riesgo}**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Nivel de riesgo",  riesgo)
    col_b.metric("Cluster asignado", str(cluster))
    col_c.metric("IMC ingresado",    f"{bmi:.1f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. RECOMENDACIONES GENERADAS CON API (Groq)
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("💡 Recomendaciones actuariales")

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    if api_key:
        prompt = f"""
Actúa como analista actuarial senior.

Explica brevemente el resultado del modelo y brinda 3 recomendaciones
prudentes, claras y profesionales basadas en el perfil del cliente.

Datos del cliente:
- Edad: {age} años
- Sexo: {sex}
- IMC: {bmi}
- Número de hijos: {children}
- Fumador: {smoker}
- Región: {region}
- Cargos médicos estimados: ${charges:,.0f}

Resultado del modelo:
- Cluster: {cluster}
- Nivel de riesgo actuarial: {riesgo}

Responde en español. Sé concreto y profesional.
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
                max_tokens=600,
            )
            st.info(completion.choices[0].message.content)
        except Exception as e:
            st.warning(f"No se pudo generar recomendación con Groq: {e}")
    else:
        rec = {
            "Bajo": (
                "✅ **Prima competitiva:** El perfil es de bajo riesgo; se puede ofrecer tarifa estándar o con descuento.\n\n"
                "✅ **Cobertura completa:** No se requieren exclusiones adicionales.\n\n"
                "✅ **Programa de fidelización:** Incentivos por renovación para mantener este perfil.\n\n"
                "✅ **Revisión anual:** Chequeo preventivo para sostener el nivel bajo."
            ),
            "Medio": (
                "⚠️ **Sobrecosto moderado:** Se recomienda prima estándar con recargo del 10–15%.\n\n"
                "⚠️ **Programa preventivo:** Inscribir en control de peso o monitoreo de IMC.\n\n"
                "⚠️ **Seguimiento semestral:** Revisión de indicadores de salud.\n\n"
                "⚠️ **Incentivo de mejora:** Descuento si reduce factores de riesgo en 6 meses."
            ),
            "Alto": (
                "🔴 **Prima diferenciada:** Recargo actuarial del 30–50% sobre tarifa base.\n\n"
                "🔴 **Examen médico previo:** Obligatorio antes de emitir la póliza.\n\n"
                "🔴 **Cláusulas de coaseguro:** Copago en siniestros de alto costo.\n\n"
                "🔴 **Plan de cesación:** Si fuma, condición contractual para mejorar condiciones."
            ),
        }
        st.info(rec.get(riesgo, "Sin recomendaciones disponibles."))
        st.caption("Para recomendaciones con IA, agrega `GROQ_API_KEY` en los secretos de Streamlit.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. VISUALIZACIÓN DE DATOS
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Visualización del perfil del cliente")

    tab1, tab2, tab3 = st.tabs(["Comparativa del grupo", "Distribución de cargos", "Tabla del dataset"])

    with tab1:
        grupo = df.copy()
        grupo["cluster"] = modelo.predict(
            grupo[["age", "bmi", "children", "charges", "sex", "smoker", "region"]]
        )
        grupo["riesgo"] = grupo["cluster"].map(mapa)
        grupo_filtrado = grupo[grupo["riesgo"] == riesgo]

        resumen = grupo_filtrado[["age", "bmi", "children", "charges"]].agg(
            ["mean", "min", "max"]
        ).T.round(2)
        resumen.columns = ["Promedio del grupo", "Mínimo", "Máximo"]
        resumen["Tu valor"] = [age, bmi, children, charges]

        st.write(f"Comparativa: **tú vs. grupo {riesgo}** ({len(grupo_filtrado):,} clientes)")
        st.dataframe(resumen, use_container_width=True)

    with tab2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colores = {"Bajo": "#2ecc71", "Medio": "#f39c12", "Alto": "#e74c3c"}
        for nivel, color in colores.items():
            datos = grupo[grupo["riesgo"] == nivel]["charges"]
            ax.hist(datos, bins=35, alpha=0.6, color=color, label=nivel, edgecolor="white")
        ax.axvline(charges, color="black", linewidth=2, linestyle="--", label=f"Tu valor: ${charges:,.0f}")
        ax.set_title("Distribución de cargos médicos por nivel de riesgo")
        ax.set_xlabel("Cargos médicos ($)")
        ax.set_ylabel("Frecuencia")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.write("Primeros 30 registros del dataset")
        st.dataframe(
            df.head(30)[["age", "sex", "bmi", "children", "smoker", "region", "charges"]],
            use_container_width=True,
        )

else:
    st.divider()
    st.subheader("📂 Vista rápida del dataset")
    st.dataframe(df.head(20), use_container_width=True)
