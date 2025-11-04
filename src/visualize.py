# Funciones de visualización: forma de onda, espectro (FFT) y espectrograma.

import numpy as np
import librosa

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
EPS = 1e-12

def waveform(y, sr, ax):
    ax.clear()
    t = np.arange(len(y)) / sr
    ax.plot(t, y)
    ax.set_title("Forma de onda (dominio del tiempo)")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")

def spectrum_fft(y, sr, ax):
    ax.clear()
    n = len(y)
    if n == 0 or sr is None or sr <= 0:
        ax.set_title("Espectro (FFT)")
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel("Magnitud [dB]")
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return

    Y = np.fft.rfft(y)
    mag = np.abs(Y) / max(n, 1)
    mag_db = 20 * np.log10(np.maximum(mag, EPS))
    freqs = np.fft.rfftfreq(n, d=1.0/sr)

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    mag_db_pos = mag_db[pos_mask]

    if freqs_pos.size:
        ax.semilogx(freqs_pos, mag_db_pos, color="tab:blue", linewidth=1.5)

    low = max(freqs_pos.min(), 1.0) if freqs_pos.size else 1.0
    high = max(sr / 2, low * 1.01)
    ax.set_xlim(low, high)
    ax.set_title("Transformada de Fourier - Magnitud (dB)")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Magnitud [dB]")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    peaks = _dominant_peaks(freqs, mag, top_n=3)
    if peaks:
        label_lines = []
        for idx, (freq, amplitude) in enumerate(peaks, start=1):
            if freq <= 0:
                continue
            ax.axvline(freq, color="tab:red", linestyle="--", alpha=0.5)
            if freqs_pos.size:
                y_val = np.interp(freq, freqs_pos, mag_db_pos)
            else:
                y_val = 20 * np.log10(max(amplitude, EPS))
            note = frequency_to_note_name(freq)
            label_lines.append(f"p{idx}: {freq:.1f} Hz ({note})")

        if label_lines:
            ax.text(
                0.98,
                0.98,
                "\n".join(label_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="tab:red",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="tab:red"),
            )


def spectrum_phase(y, sr, ax):
    ax.clear()
    n = len(y)
    if n == 0 or sr is None or sr <= 0:
        ax.set_title("Fase FFT")
        ax.set_xlabel("Frecuencia [Hz]")
        ax.set_ylabel("Fase [°]")
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return

    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    phase = np.angle(Y)
    phase_deg = np.degrees(phase)

    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    phase_deg = phase_deg[pos_mask]

    if freqs_pos.size:
        ax.semilogx(freqs_pos, phase_deg, color="tab:purple", linewidth=1)

    low = max(freqs_pos.min(), 1.0) if freqs_pos.size else 1.0
    high = max(sr / 2, low * 1.01)
    ax.set_xlim(low, high)
    ax.set_ylim(-180, 180)
    ax.set_title("Transformada de Fourier - Fase")
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Fase [°]")
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    peaks = _dominant_peaks(freqs, np.abs(Y), top_n=4)
    for freq, _ in peaks:
        if freq <= 0:
            continue
        ax.axvline(freq, color="tab:red", linestyle=":", alpha=0.3)

def spectrogram(y, sr, ax, n_fft=2048, hop_length=512):
    if hasattr(ax, "_colorbar") and ax._colorbar:
        ax._colorbar.remove()
        ax._colorbar = None

    ax.clear()
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = ax.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=S_db.max() - 80,
        vmax=S_db.max(),
    )
    ax.set_title("Espectrograma (STFT, dB)")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Frecuencia [Hz]")

    if S_db.shape[1] > 1:
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft)
        xticks = np.linspace(0, len(times) - 1, num=min(6, len(times))).astype(int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{times[i]:.2f}" for i in xticks])

    if S_db.shape[0] > 1:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        yticks = np.linspace(0, len(freqs) - 1, num=6).astype(int)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{freqs[i]:.0f}" for i in yticks])

    cbar = ax.figure.colorbar(img, ax=ax, pad=0.01, format="%+2.0f dB")
    ax._colorbar = cbar

    return img


def analyze_audio(y, sr, top_n=3):
    """Regresa métricas clave para resumir el análisis espectral."""
    n = len(y)
    duration = n / sr if sr else 0
    rms = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0

    Y = np.fft.rfft(y)
    mag = np.abs(Y) / n if n else np.array([])
    freqs = np.fft.rfftfreq(n, d=1.0/sr) if n else np.array([])
    peaks = _dominant_peaks(freqs, mag, top_n=top_n)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean() if len(y) else 0.0
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean() if len(y) else 0.0

    dominant_freqs = [freq for freq, _ in peaks]
    dominant_notes = [frequency_to_note_name(freq) for freq in dominant_freqs]

    return {
        "duration": duration,
        "rms": rms,
        "dominant_freqs": dominant_freqs,
        "dominant_notes": dominant_notes,
        "spectral_centroid": float(centroid),
        "spectral_bandwidth": float(bandwidth),
    }


def frequency_to_note_name(freq):
    if freq is None or freq <= 0:
        return "-"
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    octave = midi // 12 - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def _dominant_peaks(freqs, magnitudes, top_n=3):
    if freqs is None or magnitudes is None or len(freqs) == 0:
        return []
    pos_mask = freqs > 0
    if not np.any(pos_mask):
        return []
    freqs_pos = freqs[pos_mask]
    mags_pos = magnitudes[pos_mask]
    if mags_pos.size == 0:
        return []
    top_idx = np.argsort(mags_pos)[-top_n:][::-1]
    peaks = [(float(freqs_pos[i]), float(mags_pos[i])) for i in top_idx if freqs_pos[i] > 0]
    return peaks
