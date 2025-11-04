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

    result = {
        "features": row,
        "sample_rate": sr,
    }

    pred = model.predict(X)[0]
    result["label"] = pred

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            ranked = list(zip(classes, proba))
            ranked.sort(key=lambda item: item[1], reverse=True)
            result["confidence"] = float(ranked[0][1]) if ranked else None
            result["top_classes"] = [(str(cls), float(p)) for cls, p in ranked[:3]]
    else:
        # Para modelos sin predict_proba, intenta usar distancias KNN como proxy
        if hasattr(model, "kneighbors"):
            distances, indices = model.kneighbors(X, n_neighbors=getattr(model, "n_neighbors", 3))
            distances = distances[0]
            weights = 1 / np.maximum(distances, 1e-9)
            weights = weights / weights.sum()
            result["neigh_weights"] = weights.tolist()
            result["confidence"] = float(weights.max()) if weights.size else None

    return result
