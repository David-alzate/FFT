# Entrenamiento de un modelo clásico (KNN o RandomForest) para reconocimiento de instrumento.
# Recorre subcarpetas en data_dir y asume que cada subcarpeta corresponde a un instrumento/clase.

import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from .features import load_audio, extract_features, features_to_dataframe

def iter_audio_files(root_dir: str, exts=(".wav", ".wave")) -> List[Tuple[str, str]]:
    """
    Itera (filepath, label) asumiendo que label es el nombre de la subcarpeta inmediata.
    Ignora una carpeta llamada 'test'.
    """
    samples = []
    for sub in os.listdir(root_dir):
        full = os.path.join(root_dir, sub)
        if not os.path.isdir(full) or sub.lower() == "test":
            continue
        label = sub
        for fname in os.listdir(full):
            if fname.lower().endswith(exts):
                samples.append((os.path.join(full, fname), label))
    return samples

def build_dataset(root_dir: str):
    records = []
    for path, label in iter_audio_files(root_dir):
        try:
            y, sr = load_audio(path)
            feats = extract_features(y, sr)
            feats["label"] = label
            feats["path"] = path
            records.append(feats)
        except Exception as e:
            print(f"[WARN] No se pudo procesar {path}: {e}")
    return records

def train_and_save(root_dir: str, model_path: str, model_type: str = "knn", **kwargs):
    """
    Entrena un modelo ('knn' o 'rf') y guarda a disco en model_path.
    kwargs se pasa al constructor del modelo.
    """
    records = build_dataset(root_dir)
    if not records:
        raise RuntimeError("No se encontraron muestras en las subcarpetas de datos.")

    df = features_to_dataframe(records)
    X = df.drop(columns=["label", "path"]).values
    y = df["label"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

    if model_type.lower() == "knn":
        n_neighbors = kwargs.pop("n_neighbors", 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    elif model_type.lower() == "rf":
        n_estimators = kwargs.pop("n_estimators", 200)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, **kwargs)
    else:
        raise ValueError("model_type debe ser 'knn' o 'rf'")

    model.fit(Xtr, ytr)
    # Intento de validación (si hay variedad de clases)
    acc = -1.0
    try:
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)
        print("Accuracy de validación:", acc)
        print(classification_report(yte, ypred))
    except Exception:
        pass

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "columns": df.drop(columns=["label", "path"]).columns.tolist()}, f)

    return {"num_samples": len(df), "classes": sorted(set(y)), "val_accuracy": float(acc)}