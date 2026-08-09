"""
AgroDetect - Detección de Enfermedades en Hojas de Café
==========================================================
Servicio Web (Streamlit) que permite:
  1. Cargar/capturar una imagen de una hoja de café.
  2. Detectar la enfermedad/plaga presente usando un modelo de IA (TensorFlow),
     entrenado en Google Colab.
  3. Mostrar el porcentaje de confianza de la predicción.
  4. Generar automáticamente, vía la API de Groq (LLM), una orientación técnica:
     descripción, recomendaciones de manejo preventivo, buenas prácticas y
     acciones de seguimiento.
  5. Descargar un reporte en PDF con el diagnóstico.

Desplegable en Streamlit Community Cloud (Computación en la Nube).
"""

import io
from datetime import datetime

import streamlit as st
from PIL import Image

from utils.model_utils import DISEASE_INFO, get_model
from utils.groq_utils import generate_recommendation
from utils.pdf_utils import build_pdf_report

# ----------------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroDetect | Claudia Aguilar",
    page_icon="🍃🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background-color: #F3EFE6;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3, .agro-title {
        font-family: Georgia, 'Times New Roman', serif;
        color: #2B2118;
    }
    .agro-card {
        background-color: #FFFFFF;
        border: 1px solid #E4DCC9;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .agro-subtitle {
        color: #6B6152;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }
    .agro-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: .03em;
    }
    .confidence-box {
        text-align: right;
    }
    .confidence-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2B2118;
        font-family: Georgia, serif;
        line-height: 1;
    }
    .confidence-label {
        color: #8A8072;
        font-size: 0.75rem;
        letter-spacing: .05em;
        text-transform: uppercase;
    }
    .section-title {
        font-weight: 700;
        color: #2B2118;
        font-size: 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }
    .section-num {
        display:inline-flex; align-items:center; justify-content:center;
        width: 22px; height: 22px; border-radius: 50%;
        background:#2F4A34; color:#fff; font-size:0.72rem; margin-right:8px;
    }
    .history-item {
        display:flex; justify-content:space-between; padding:6px 0;
        border-bottom: 1px solid #EFEAE0; font-size: 0.85rem; color:#4A4234;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #FAF8F2;
        border: 1.5px dashed #C9BFA6;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2F4A34;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #24382A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Estado de sesión

if "history" not in st.session_state:
    st.session_state.history = [] 
if "last_result" not in st.session_state:
    st.session_state.last_result = None  

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None


st.markdown("<h2 class='agro-title'>🍃 AgroDetect &nbsp;|&nbsp; Diagnóstico Foliar de Café | Claudia </h2>", unsafe_allow_html=True)
st.markdown(
    "<div class='agro-subtitle'>Servicio web de Computación en la Nube para la detección temprana de "
    "enfermedades y plagas en hojas de café mediante Inteligencia Artificial.</div>",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1, 1.15], gap="large")

# ----------------------------------------------------------------------------
# COLUMNA IZQUIERDA: Captura de imagen
# ----------------------------------------------------------------------------
with col_left:
    st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.15rem;'>📷 Captura de Imagen Foliar</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='agro-subtitle'>Posiciona la hoja de café bajo luz natural. "
        "El sistema detectará automáticamente signos de Roya, Cercospora/Phoma o Minador.</div>",
        unsafe_allow_html=True,
    )

    mode = st.radio("Fuente de imagen", ["Subir archivo", "Usar cámara"], horizontal=True, label_visibility="collapsed")

    uploaded_image = None
    if mode == "Subir archivo":
        file = st.file_uploader("Sube una foto de la hoja", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if file is not None:
            uploaded_image = Image.open(file)
            st.session_state["_current_filename"] = file.name
    else:
        cam_file = st.camera_input("Captura con la cámara", label_visibility="collapsed")
        if cam_file is not None:
            uploaded_image = Image.open(cam_file)
            st.session_state["_current_filename"] = "captura_camara.jpg"

    if uploaded_image is not None:
        st.image(uploaded_image, use_container_width=True)
        analyze = st.button("🔍 Analizar hoja", use_container_width=True, type="primary")
    else:
        st.info("Sube una imagen o usa la cámara para comenzar el diagnóstico.")
        analyze = False

    st.markdown("</div>", unsafe_allow_html=True)

    # Historial reciente
    if st.session_state.history:
        st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🕘 Historial reciente</div>", unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-6:]):
            st.markdown(
                f"<div class='history-item'><span>🟢 {h['label']}</span>"
                f"<span>{h['timestamp']}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Lógica de inferencia + recomendación (se ejecuta al presionar "Analizar")
# ----------------------------------------------------------------------------
if uploaded_image is not None and analyze:
    with st.spinner("Analizando imagen con el modelo de IA..."):
        model = get_model()
        pred_class, confidence, distribution = model.predict(uploaded_image)
        info = DISEASE_INFO.get(pred_class, DISEASE_INFO["Sano"])

    with st.spinner("Generando orientación técnica con la API de Groq..."):
        recommendation = generate_recommendation(
            disease_key=pred_class,
            display_name=info["display_name"],
            confidence=confidence,
            api_key=GROQ_API_KEY,
        )

    img_bytes_io = io.BytesIO()
    uploaded_image.convert("RGB").save(img_bytes_io, format="JPEG")

    st.session_state.last_result = {
        "pred_class": pred_class,
        "info": info,
        "confidence": confidence,
        "distribution": distribution,
        "recommendation": recommendation,
        "image_bytes": img_bytes_io.getvalue(),
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
    }
    st.session_state.history.append({
        "label": info["display_name"],
        "timestamp": datetime.now().strftime("%d/%m %H:%M"),
    })

# ----------------------------------------------------------------------------
# COLUMNA DERECHA: Resultado del diagnóstico
# ----------------------------------------------------------------------------
with col_right:
    result = st.session_state.last_result
    st.markdown("<div class='agro-card'>", unsafe_allow_html=True)

    if result is None:
        st.markdown("<div class='section-title' style='font-size:1.15rem;'>🩺 Diagnóstico</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='agro-subtitle'>Aquí aparecerá el resultado del análisis: enfermedad detectada, "
            "porcentaje de confianza y una guía técnica de manejo generada automáticamente.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        info = result["info"]
        rec = result["recommendation"]
        header_l, header_r = st.columns([2, 1])
        with header_l:
            st.markdown(f"<div class='agro-subtitle'>ÚLTIMO DIAGNÓSTICO &nbsp;·&nbsp; {result['timestamp']}</div>", unsafe_allow_html=True)
            st.markdown(f"<h3 class='agro-title' style='margin:0;'>{info['display_name']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='agro-subtitle' style='margin-top:2px;'><i>{info['scientific_name']}</i></div>", unsafe_allow_html=True)
        with header_r:
            st.markdown(
                f"<div class='confidence-box'><div class='confidence-value'>{result['confidence']*100:.1f}%</div>"
                f"<div class='confidence-label'>Confianza</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#EFEAE0;'>", unsafe_allow_html=True)
        st.caption(f"Fuente de la orientación técnica: {rec.get('_source', 'groq')}")

        st.markdown(f"<div class='section-title'><span class='section-num'>01</span>Diferenciación a simple vista</div>", unsafe_allow_html=True)
        st.write(rec.get("diferenciacion", ""))

        st.markdown(f"<div class='section-title'><span class='section-num'>02</span>Manejo agronómico preventivo y correctivo</div>", unsafe_allow_html=True)
        for item in rec.get("manejo_preventivo", []):
            st.markdown(f"- {item}")

        st.markdown(f"<div class='section-title'><span class='section-num'>03</span>Buenas prácticas</div>", unsafe_allow_html=True)
        for item in rec.get("buenas_practicas", []):
            st.markdown(f"- {item}")

        st.markdown(f"<div class='section-title'><span class='section-num'>04</span>Monitoreo y seguimiento</div>", unsafe_allow_html=True)
        for item in rec.get("monitoreo_seguimiento", []):
            st.markdown(f"- {item}")

        st.markdown(f"<div class='section-title'><span class='section-num'>05</span>Consulta a un técnico</div>", unsafe_allow_html=True)
        st.write(rec.get("cuando_consultar_tecnico", ""))

        st.markdown(f"<div class='section-title'><span class='section-num'>06</span>Registro y trazabilidad</div>", unsafe_allow_html=True)
        st.write(rec.get("registro_trazabilidad", ""))

        with st.expander("Ver distribución de probabilidades del modelo"):
            for cls, p in sorted(result["distribution"].items(), key=lambda x: -x[1]):
                label = DISEASE_INFO.get(cls, {}).get("display_name", cls)
                st.progress(p, text=f"{label}: {p*100:.1f}%")

        pdf_bytes = build_pdf_report(
            image_bytes=result["image_bytes"],
            display_name=info["display_name"],
            scientific_name=info["scientific_name"],
            confidence=result["confidence"],
            recommendation=rec,
        )
        st.download_button(
            "📄 Descargar reporte en PDF",
            data=pdf_bytes,
            file_name=f"diagnostico_{result['pred_class'].lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center; color:#9B927E; font-size:0.78rem; margin-top:1.5rem;'>"
    "© 2026 AgroDetect · Soporte HICAFE · Proyecto académico de Computación en la Nube- By: Claudia Aguilar</div>",
    unsafe_allow_html=True,
)
