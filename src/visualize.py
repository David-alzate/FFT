# Funciones de visualización: forma de onda, espectro (FFT) y espectrograma.

import numpy as np
import librosa

def waveform(y, sr, ax):
    ax.clear()
    t = np.arange(len(y)) / sr
    ax.plot(t, y)
    ax.set_title("Forma de onda")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")

def spectrum_fft(y, sr, ax):
    ax.clear()
    n = len(y)
    Y = np.fft.rfft(y)
    mag = np.abs(Y) / n
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    ax.plot(freqs, mag)
    ax.set_xlim(0, sr/2)
    ax.set_title("Espectro (FFT)")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("|X(f)|")

def spectrogram(y, sr, ax, n_fft=2048, hop_length=512):
    ax.clear()
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = ax.imshow(S_db, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title("Espectrograma (dB)")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frecuencias")
    return img