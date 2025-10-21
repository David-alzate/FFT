# Carga el modelo entrenado y predice el instrumento de un nuevo audio.

import pickle
import numpy as np
from .features import load_audio, extract_features

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["columns"]

def predict_file(audio_path: str, model_path: str):
    model, columns = load_model(model_path)
    y, sr = load_audio(audio_path)
    feats = extract_features(y, sr)
    # Asegurar el orden de columnas
    row = {c: feats.get(c, 0.0) for c in columns}
    X = np.array([list(row.values())])
    pred = model.predict(X)[0]
    return pred