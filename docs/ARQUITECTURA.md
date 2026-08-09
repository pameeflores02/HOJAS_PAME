# Documentación Técnica — AgroDetect

## 1. Objetivo del sistema
Detectar, mediante inteligencia artificial, enfermedades y plagas en hojas de café a partir de una
fotografía, y entregar al caficultor una orientación técnica de manejo generada automáticamente,
todo a través de un Servicio Web desplegado en la nube.

## 2. Arquitectura del sistema

### 2.1 Vista general
El sistema sigue una arquitectura de **aplicación web monolítica desplegada en la nube (PaaS)**,
con tres componentes de inteligencia integrados por composición (no microservicios separados, dado
el alcance académico del proyecto):

1. **Capa de presentación / lógica de aplicación** — Streamlit (`app.py`), que sirve tanto la
   interfaz web como la orquestación de las llamadas a los modelos de IA.
2. **Modelo de visión artificial** — Red neuronal convolucional (TensorFlow/Keras) entrenada con
   *transfer learning* sobre MobileNetV2, cargada localmente dentro del contenedor de la app.
3. **Modelo de lenguaje (LLM)** — API de Groq, consumida vía HTTPS, para generar el contenido
   textual de recomendaciones a partir del diagnóstico visual.

### 2.2 Diagrama de flujo
```
Usuario                Streamlit App                 Modelo TF (local)        API Groq (nube)
  │  sube/captura foto        │                              │                        │
  ├──────────────────────────▶│                              │                        │
  │                            │  preprocesa imagen (128x128) │                        │
  │                            ├─────────────────────────────▶│                        │
  │                            │  clase + % confianza          │                        │
  │                            │◀─────────────────────────────┤                        │
  │                            │  prompt (clase + confianza)   │                        │
  │                            ├───────────────────────────────────────────────────────▶│
  │                            │  JSON: descripción, manejo preventivo, buenas prácticas,│
  │                            │  monitoreo, consulta a técnico, trazabilidad            │
  │                            │◀───────────────────────────────────────────────────────┤
  │  UI con diagnóstico +      │                              │                        │
  │  recomendaciones + PDF     │                              │                        │
  │◀───────────────────────────┤                              │                        │
```

## 3. Servicios en la nube utilizados
| Servicio | Rol | Notas |
|---|---|---|
| **Google Colab** | Entrenamiento del modelo de visión artificial (GPU gratuita). | Ver `notebooks/entrenamiento_modelo_colab.ipynb`. |
| **Streamlit Community Cloud** | Hosting/despliegue del Servicio Web (PaaS, computación en la nube). | Build automático desde GitHub + `requirements.txt`. |
| **API de Groq** | Inferencia de LLM (`llama-3.3-70b-versatile`) para generar la orientación técnica en tiempo real. | Consumida vía HTTPS con streaming/JSON estructurado. |
| **GitHub** | Control de versiones y CI de despliegue (integración con Streamlit Cloud). | Repositorio público del proyecto. |

## 4. Tecnologías empleadas
- **Python 3** — lenguaje principal.
- **TensorFlow / Keras** — entrenamiento e inferencia del modelo de visión artificial (CNN,
  transfer learning con MobileNetV2).
- **Streamlit** — framework para construir y desplegar la interfaz web e interactuar con el usuario.
- **Groq (SDK oficial `groq`)** — generación de texto (LLM) para las recomendaciones técnicas.
- **Pillow (PIL)** — manipulación y preprocesamiento de imágenes.
- **ReportLab** — generación de reportes PDF descargables.
- **NumPy** — operaciones numéricas sobre las predicciones del modelo.

## 5. Flujo de funcionamiento del sistema
1. El usuario abre la aplicación web (desplegada en Streamlit Community Cloud).
2. Sube una foto de una hoja de café o la captura con la cámara del dispositivo.
3. Al presionar **"Analizar hoja"**:
   a. La imagen se redimensiona a 128×128 px y se normaliza.
   b. El modelo TensorFlow (cargado en memoria del contenedor) predice la clase
      (`Roya`, `Phoma`, `Minador` o `Sano`) y su probabilidad.
   c. Se construye un *prompt* estructurado con la clase detectada y el porcentaje de confianza.
   d. Se llama a la API de Groq solicitando una respuesta en formato JSON con las secciones:
      descripción, diferenciación a simple vista, manejo preventivo/correctivo, buenas prácticas,
      monitoreo y seguimiento, cuándo consultar a un técnico, y registro/trazabilidad.
   e. Si la llamada a Groq falla o no hay API key configurada, se utiliza un conjunto de
      recomendaciones de respaldo (offline) equivalente en estructura, para que el sistema nunca
      quede sin orientación técnica.
4. El resultado se muestra en pantalla (diagnóstico + confianza + recomendaciones) y se agrega al
   historial de la sesión.
5. El usuario puede descargar un **reporte PDF** con toda la información generada, mediante ReportLab.

## 6. Modelo de datos / dataset
El modelo fue entrenado con ~1,800 imágenes distribuidas en 4 clases balanceadas
(Roya: 400, Phoma: 499, Minador: 500, Sano: 400), provenientes del `Dataset.zip` entregado para el
proyecto. Ver `notebooks/entrenamiento_modelo_colab.ipynb` para el detalle completo del
preprocesamiento, la arquitectura y las métricas de entrenamiento (exactitud de validación,
matriz de confusión, reporte de clasificación).

## 7. Limitaciones conocidas y trabajo futuro
- El dataset entregado no incluía imágenes reales para las clases *Cercospora* (estudio fotográfico)
  y *Ácaro/araña roja*, ya que esos archivos eran punteros de Git LFS sin contenido descargado; el
  sistema actual cubre 4 clases (Roya, Phoma, Minador, Sano). Se recomienda ampliar el dataset con
  dichas clases para una cobertura más completa.
- El modelo de referencia entrenado en este repositorio corresponde a una CNN base entrenada desde
  cero como línea base funcional; se recomienda ejecutar el notebook de Colab (transfer learning
  con MobileNetV2 + fine-tuning) para obtener una exactitud de producción más alta antes de la
  entrega final.
- Como mejora futura, se podría desplegar el modelo como un microservicio independiente (p. ej.
  FastAPI + TensorFlow Serving) para desacoplar el cómputo de inferencia de la capa de presentación.
