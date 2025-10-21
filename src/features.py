# Herramientas de extracción de características espectrales y cálculo de FFT.
# - Carga de audio con librosa (mono).
# - FFT con numpy.fft.rfft para visualizar el espectro.
# - Features: spectral_centroid, spectral_bandwidth, spectral_rolloff, MFCCs.

import numpy as np
import librosa
import pandas as pd

def load_audio(path, sr=22050):
    """Carga un archivo de audio como mono y devuelve (y, sr)."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def compute_fft(y, sr):
    """
    Calcula la FFT de una señal 1D.
    Retorna frecuencias (Hz) y magnitudes normalizadas.
    """
    n = len(y)
    # FFT unilateral (positiva) para señal real
    Y = np.fft.rfft(y)
    mag = np.abs(Y) / n
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    return freqs, mag

def extract_features(y, sr, n_mfcc=13):
    """
    Extrae un vector de características robusto y compacto (agregando estadísticas).
    Devuelve un diccionario {feature_name: value}.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    def stats(name, arr):
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_min": float(np.min(arr)),
            f"{name}_max": float(np.max(arr)),
        }

    feats = {}
    feats.update(stats("centroid", centroid))
    feats.update(stats("bandwidth", bandwidth))
    feats.update(stats("rolloff", rolloff))

    # MFCCs: agregamos mean y std por coeficiente
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfcc[i]))

    return feats

def features_to_dataframe(records):
    """Convierte una lista de dicts de características en DataFrame, asegurando columnas consistentes."""
    return pd.DataFrame.from_records(records).fillna(0.0)