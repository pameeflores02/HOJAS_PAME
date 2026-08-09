"""
model_utils.py
----------------
Carga del modelo de Deep Learning (entrenado en Google Colab con TensorFlow/Keras)
y funciones de preprocesamiento / inferencia para la clasificación de
enfermedades en hojas de café.
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "coffee_leaf_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = (128, 128)

# Metadatos de cada clase para mostrar en la interfaz (nombre visible,
# color de acento y descripción corta). Deben coincidir con las carpetas
# usadas durante el entrenamiento en Colab.
DISEASE_INFO = {
    "Roya": {
        "display_name": "Roya del Café",
        "scientific_name": "Hemileia vastatrix",
        "color": "#B5651D",
        "severity": "alta",
    },
    "Phoma": {
        "display_name": "Phoma / Mancha de Phoma",
        "scientific_name": "Phoma spp.",
        "color": "#7A4B8C",
        "severity": "media",
    },
    "Minador": {
        "display_name": "Minador de la Hoja",
        "scientific_name": "Leucoptera coffeella",
        "color": "#C9A227",
        "severity": "media",
    },
    "Sano": {
        "display_name": "Hoja Sana",
        "scientific_name": "Sin patógeno detectado",
        "color": "#3E7C3E",
        "severity": "ninguna",
    },
}


def load_class_names() -> list:
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@tf.function(reduce_retracing=True)
def _predict_tensor(model, x):
    return model(x, training=False)


class LeafDiseaseModel:
    """Envuelve el modelo Keras entrenado y expone una API simple de inferencia."""

    def __init__(self, model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
        self.model = tf.keras.models.load_model(model_path)
        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(IMG_SIZE)
        arr = np.array(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # el modelo ya incluye la capa Rescaling(1./255)
        return arr

    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        """Devuelve (clase_predicha, confianza[0-1], distribución completa por clase)."""
        arr = self.preprocess(image)
        probs = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred_class = self.class_names[idx]
        confidence = float(probs[idx])
        distribution = {cls: float(p) for cls, p in zip(self.class_names, probs)}
        return pred_class, confidence, distribution


_model_singleton = None


def get_model() -> LeafDiseaseModel:
    """Carga el modelo una sola vez (cache a nivel de proceso)."""
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = LeafDiseaseModel()
    return _model_singleton
