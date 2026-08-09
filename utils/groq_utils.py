"""
groq_utils.py
--------------
Integración con la API de Groq (LLM) para generar, en tiempo real, una
orientación técnica personalizada a partir del diagnóstico entregado por el
modelo de visión artificial: descripción de la enfermedad, recomendaciones de
manejo preventivo/correctivo, buenas prácticas y acciones de seguimiento.
"""

import json
import os
from typing import Dict, List

from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Eres un ingeniero agrónomo experto en fitopatología del café (Coffea arabica), "
    "especializado en el manejo integrado de plagas y enfermedades en Centroamérica. "
    "Respondes siempre en español, de forma técnica pero clara, dirigida a un caficultor "
    "o técnico de campo. Debes responder EXCLUSIVAMENTE en formato JSON válido, sin texto "
    "adicional antes o después, siguiendo exactamente el esquema solicitado."
)

JSON_SCHEMA_INSTRUCTIONS = """
Devuelve un JSON con esta estructura exacta (sin comentarios, sin markdown):
{
  "descripcion": "2-3 frases explicando qué es la enfermedad/plaga y cómo se manifiesta en la hoja",
  "diferenciacion": "1-2 frases sobre cómo distinguirla a simple vista de otras afecciones similares",
  "manejo_preventivo": ["acción 1", "acción 2", "acción 3", "acción 4"],
  "buenas_practicas": ["práctica 1", "práctica 2", "práctica 3"],
  "monitoreo_seguimiento": ["acción 1", "acción 2", "acción 3"],
  "cuando_consultar_tecnico": "1-2 frases indicando en qué casos se debe escalar a un técnico HICAFE/asesor agrícola",
  "registro_trazabilidad": "1-2 frases sobre qué datos conviene registrar para dar seguimiento al lote/parcela"
}
"""


def _fallback_recommendation(disease_key: str) -> Dict:
    """Contenido de respaldo (sin conexión a Groq) para que la app nunca se quede
    sin recomendaciones, por ejemplo si no hay API key configurada o falla la red."""
    fallback = {
        "Roya": {
            "descripcion": "La roya del café, causada por el hongo Hemileia vastatrix, produce manchas "
                            "amarillo-anaranjadas polvorientas en el envés de la hoja, que luego se necrosan "
                            "y provocan defoliación prematura.",
            "diferenciacion": "A diferencia de otras manchas foliares, la roya presenta polvo anaranjado "
                               "(uredosporas) visible al frotar el envés de la hoja.",
            "manejo_preventivo": [
                "Aplicar fungicidas cúpricos o triazoles según calendario preventivo antes de lluvias.",
                "Mantener sombra regulada (40-50%) para reducir humedad foliar.",
                "Fertilizar de forma balanceada, evitando exceso de nitrógeno.",
                "Podar y renovar tejido productivo en lotes con alta incidencia histórica.",
            ],
            "buenas_practicas": [
                "Usar variedades resistentes en resiembras (Catimor, Costa Rica 95, IHCAFE90).",
                "Evitar el estrés hídrico de la planta.",
                "Eliminar restos de hojas caídas infectadas.",
            ],
            "monitoreo_seguimiento": [
                "Muestrear el 10% de las plantas por lote cada 15 días en época lluviosa.",
                "Registrar el porcentaje de hojas afectadas por planta.",
                "Revisar efectividad de la aplicación fungicida a los 7-10 días.",
            ],
            "cuando_consultar_tecnico": "Si la incidencia supera el 30% del follaje o persiste tras dos "
                                        "aplicaciones fungicidas, se recomienda consultar a un técnico agrícola.",
            "registro_trazabilidad": "Registrar fecha, producto y dosis de cada aplicación, así como el "
                                      "porcentaje de incidencia por parcela para ajustar el manejo integral.",
        },
        "Phoma": {
            "descripcion": "Cercospora / Mancha de Phoma produce manchas circulares de 3-8 mm con centro "
                            "grisáceo-necrótico y halo amarillo-anaranjado, asociada a estrés hídrico y "
                            "deficiencias nutricionales.",
            "diferenciacion": "A diferencia de la roya, no hay pústulas ni polvo en el envés; se confunde "
                               "con manchas de nutrición (deficiencia de Mn), pero estas carecen de halo definido.",
            "manejo_preventivo": [
                "Aumentar fertilización nitrogenada (urea foliar al 2%) y potásica.",
                "Regular sombra al 40-50% para reducir estrés hídrico.",
                "Aplicar caldo bordelés preventivo antes de lluvias intensas.",
                "Evitar trabajos en campo con follaje mojado para no diseminar esporas.",
            ],
            "buenas_practicas": [
                "Realizar análisis foliar bianual para ajustar fertilización.",
                "Mantener cobertura de sombra adecuada según la especie asociada.",
                "Evitar podas o manipulación de plantas en clima húmedo.",
            ],
            "monitoreo_seguimiento": [
                "Monitorear quincenalmente en épocas secas y calurosas (febrero-abril).",
                "Revisar hojas del tercio medio de la planta.",
                "Registrar recuperación del color verde intenso tras la corrección.",
            ],
            "cuando_consultar_tecnico": "Si las manchas aparecen en más del 30% del follaje o persisten "
                                        "tras dos aplicaciones fungicidas, consultar a un técnico para descartar "
                                        "un problema nutricional primario.",
            "registro_trazabilidad": "Documentar análisis foliar, niveles de sombra, fechas de aplicaciones y "
                                      "condiciones climáticas previas para ajustar el manejo integral del cultivo.",
        },
        "Minador": {
            "descripcion": "El minador de la hoja (Leucoptera coffeella) es una plaga cuyas larvas excavan "
                            "galerías dentro del tejido foliar, formando minas necróticas irregulares.",
            "diferenciacion": "Se reconoce por las galerías o 'minas' serpenteantes visibles al trasluz de la "
                               "hoja, a diferencia de manchas fúngicas que son circulares y superficiales.",
            "manejo_preventivo": [
                "Favorecer enemigos naturales (avispas parasitoides) evitando insecticidas de amplio espectro.",
                "Aplicar productos específicos (spinosad, abamectina) solo si el umbral económico lo justifica.",
                "Mantener sombra adecuada, ya que el minador prolifera en pleno sol.",
                "Fertilizar balanceadamente para reducir susceptibilidad de la planta.",
            ],
            "buenas_practicas": [
                "Conservar franjas de vegetación que alberguen controladores biológicos.",
                "Evitar aplicaciones calendarizadas innecesarias de insecticidas.",
                "Priorizar variedades y sistemas agroforestales con sombra diversificada.",
            ],
            "monitoreo_seguimiento": [
                "Contar hojas minadas por planta cada 15 días en época seca.",
                "Calcular el porcentaje de infestación (hojas con minas / hojas totales).",
                "Registrar presencia de parasitoides naturales en las minas.",
            ],
            "cuando_consultar_tecnico": "Si la infestación supera el 15-20% de hojas minadas o se observa "
                                        "defoliación significativa, se recomienda apoyo técnico especializado.",
            "registro_trazabilidad": "Llevar registro del porcentaje de infestación, manejo de sombra y uso de "
                                      "controles biológicos o químicos aplicados por lote.",
        },
        "Sano": {
            "descripcion": "La hoja analizada no presenta signos visibles de enfermedades ni plagas comunes "
                            "del cafeto. El tejido foliar muestra una coloración y textura normales.",
            "diferenciacion": "Ausencia de manchas, pústulas, galerías o decoloraciones anormales en el haz "
                               "y envés de la hoja.",
            "manejo_preventivo": [
                "Mantener el programa regular de fertilización balanceada.",
                "Continuar con el manejo preventivo de sombra y humedad.",
                "Sostener el monitoreo fitosanitario rutinario del lote.",
                "Evitar el estrés hídrico mediante riego o cobertura vegetal según la zona.",
            ],
            "buenas_practicas": [
                "Realizar podas sanitarias de mantenimiento.",
                "Diversificar la sombra para favorecer la biodiversidad benéfica.",
                "Llevar un calendario de fertilización y monitoreo.",
            ],
            "monitoreo_seguimiento": [
                "Inspeccionar el cultivo cada 15-30 días como buena práctica preventiva.",
                "Registrar el estado fitosanitario general del lote.",
                "Actualizar el historial de monitoreo aunque no se detecten problemas.",
            ],
            "cuando_consultar_tecnico": "No se requiere intervención inmediata; se recomienda mantener el "
                                        "monitoreo rutinario y consultar a un técnico ante cualquier cambio visible.",
            "registro_trazabilidad": "Registrar la fecha de inspección y el estado 'sano' para mantener la "
                                      "trazabilidad histórica del lote, útil para detectar cambios futuros.",
        },
    }
    return fallback.get(disease_key, fallback["Sano"])


def generate_recommendation(disease_key: str, display_name: str, confidence: float,
                             api_key: str = None) -> Dict:
    """
    Llama a la API de Groq para generar la orientación técnica personalizada.
    Si no hay API key disponible o la llamada falla, retorna contenido de respaldo
    para que la aplicación siga siendo funcional.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        result = _fallback_recommendation(disease_key)
        result["_source"] = "offline"
        return result

    try:
        client = Groq(api_key=api_key)
        user_prompt = (
            f"El sistema de visión artificial detectó: '{display_name}' "
            f"con una confianza del {confidence*100:.1f}%.\n"
            f"Genera la orientación técnica para el caficultor.\n{JSON_SCHEMA_INSTRUCTIONS}"
        )
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        result = json.loads(content)
        result["_source"] = "groq"
        return result
    except Exception as exc:  # noqa: BLE001 - queremos capturar cualquier error de red/API
        result = _fallback_recommendation(disease_key)
        result["_source"] = f"offline (error Groq: {exc.__class__.__name__})"
        return result
