# AgroDetect — Detección de Enfermedades en Hojas de Café🍃 
# Claudia Aguilar

Servicio Web basado en **Computación en la Nube** que detecta enfermedades y plagas en hojas de
café mediante un modelo de **visión artificial (TensorFlow/Keras)** entrenado en **Google Colab**,
y genera automáticamente recomendaciones técnicas de manejo preventivo mediante la **API de Groq (LLM)**.

## Índice
- [Demo](#demo)
- [Características](#características)
- [Arquitectura](#arquitectura)
- [Instalación local](#instalación-local)
- [Entrenamiento del modelo (Google Colab)](#entrenamiento-del-modelo-google-colab)
- [Configuración de la API de Groq](#configuración-de-la-api-de-groq)
- [Despliegue en Streamlit Community Cloud](#despliegue-en-streamlit-community-cloud)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Clases detectadas](#clases-detectadas)
- [Créditos](#créditos)

## Demo
🔗 URL pública: `<pega aquí la URL de Streamlit Community Cloud una vez desplegada>`

## Características
- Carga de imagen desde archivo **o** captura directa con cámara.
- Clasificación con un modelo CNN (TensorFlow) entrenado en Colab.
- Porcentaje de confianza y distribución de probabilidades por clase.
- Orientación técnica generada en tiempo real con la **API de Groq**: descripción,
  diferenciación a simple vista, manejo preventivo/correctivo, buenas prácticas,
  monitoreo y registro/trazabilidad.
- Descarga de un **reporte en PDF** con el diagnóstico completo.
- Historial de diagnósticos recientes en la sesión.
- Interfaz web amigable, responsiva, en español.

## Arquitectura
```
┌───────────────┐      imagen      ┌──────────────────────┐
│   Navegador   │ ───────────────▶ │   Streamlit App       │
│ (usuario web) │ ◀─────────────── │   (Cómputo en la Nube)│
└───────────────┘   resultado UI   └──────────┬────────────┘
                                               │
                          ┌────────────────────┼─────────────────────┐
                          ▼                    ▼                     ▼
                 ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
                 │ Modelo TF/Keras │  │   API de Groq    │  │  ReportLab (PDF) │
                 │ (coffee_leaf_   │  │  (LLM - recomen- │  │  generación de   │
                 │  model.h5)      │  │  daciones)        │  │  reporte técnico │
                 └─────────────────┘  └─────────────────┘  └──────────────────┘
```
Ver documentación ampliada en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

## Instalación local

1. Clona el repositorio y entra a la carpeta:
   ```bash
   git clone <url-de-tu-repositorio>
   cd coffee-leaf-disease-detector
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate       # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configura tu API key de Groq (ver sección siguiente).

4. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

5. Abre el navegador en `http://localhost:8501`.

> El modelo entrenado (`model/coffee_leaf_model.h5`) ya viene incluido en el repositorio, por lo
> que la app funciona de inmediato. Si quieres reentrenarlo con más datos o mejor exactitud, sigue
> la sección de Colab.

## Entrenamiento del modelo (Google Colab)

El notebook [`notebooks/entrenamiento_modelo_colab.ipynb`](notebooks/entrenamiento_modelo_colab.ipynb)
contiene el flujo completo de entrenamiento usando **transfer learning con MobileNetV2**:

1. Abre el notebook en [Google Colab](https://colab.research.google.com/).
2. Activa una GPU (`Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU`).
3. Sube el archivo `Dataset.zip` cuando el notebook lo solicite.
4. Ejecuta todas las celdas en orden. El notebook:
   - Organiza el dataset en carpetas por clase (`Roya`, `Phoma`, `Minador`, `Sano`).
   - Entrena la cabeza clasificadora sobre MobileNetV2 preentrenada en ImageNet.
   - Aplica *fine-tuning* de las últimas capas para mejorar la exactitud.
   - Genera una matriz de confusión y un reporte de clasificación.
   - Exporta `coffee_leaf_model.h5` y `class_names.json`.
5. Descarga ambos archivos y reemplázalos dentro de `model/` en este repositorio.

> **Nota sobre el dataset**: algunos archivos de `Dataset.zip` (Cercospora en estudio fotográfico,
> Ácaro/araña roja, y algunas variantes "_Prueba") son punteros de **Git LFS** y no contienen las
> imágenes reales dentro del zip entregado. El modelo incluido en este repositorio se entrenó con
> las 4 clases cuyas imágenes sí estaban disponibles: **Roya, Phoma, Minador y Sano** (~1,800
> imágenes en total). Si tu equipo tiene acceso al repositorio Git LFS original, puedes descargar
> las clases adicionales y volver a entrenar para ampliar la cobertura del modelo.

## Configuración de la API de Groq

1. Crea una cuenta gratuita en [console.groq.com](https://console.groq.com/) y genera una API key.
2. **En local**: copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y coloca tu key:
   ```toml
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   ```
3. **En Streamlit Community Cloud**: ve a tu app → *Settings* → *Secrets*, y pega el mismo contenido.

> Si no se configura ninguna API key, la aplicación **sigue funcionando** utilizando un conjunto de
> recomendaciones técnicas de respaldo (offline) para cada enfermedad, de modo que el sistema nunca
> deja al usuario sin orientación — pero para cumplir el requisito de integración obligatoria con
> Groq, la key debe estar configurada al momento de la evaluación.

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público o compartido con el docente).
2. Entra a [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con GitHub.
3. Clic en **"New app"**, selecciona el repositorio, la rama (`main`) y el archivo principal
   (`app.py`).
4. En **Advanced settings > Secrets**, pega tu `GROQ_API_KEY` (ver sección anterior).
5. Despliega. Streamlit instalará automáticamente las dependencias listadas en
   `requirements.txt`.
6. Copia la URL pública generada y agrégala en la sección [Demo](#demo) de este README.

## Estructura del repositorio
```
coffee-leaf-disease-detector/
├── app.py                     # Aplicación principal Streamlit
├── requirements.txt           # Dependencias del proyecto
├── README.md
├── .streamlit/
│   ├── config.toml            # Tema visual de la app
│   └── secrets.toml.example   # Plantilla para la API key de Groq
├── model/
│   ├── coffee_leaf_model.h5   # Modelo entrenado (TensorFlow/Keras)
│   ├── class_names.json       # Orden de las clases del modelo
│   └── training_report.json   # Métricas del último entrenamiento
├── utils/
│   ├── model_utils.py         # Carga del modelo + inferencia
│   ├── groq_utils.py          # Integración con la API de Groq
│   └── pdf_utils.py           # Generación del reporte PDF
├── notebooks/
│   └── entrenamiento_modelo_colab.ipynb   # Notebook de entrenamiento (Google Colab)
└── docs/
    └── ARQUITECTURA.md        # Documentación técnica ampliada
```

## Clases detectadas
| Clase | Nombre científico | Descripción breve |
|---|---|---|
| 🟠 Roya | *Hemileia vastatrix* | Manchas anaranjadas polvorientas en el envés de la hoja. |
| 🟣 Phoma | *Phoma* spp. | Manchas circulares grisáceo-necróticas con halo amarillo. |
| 🟡 Minador | *Leucoptera coffeella* | Galerías/minas serpenteantes dentro del tejido foliar. |
| 🟢 Sano | — | Hoja sin síntomas visibles. |

## Créditos
Claudia Aguilar
Proyecto académico — Computación en la Nube. Desarrollado con Python, TensorFlow, Streamlit,
API de Groq y ReportLab.
