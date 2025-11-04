# Interfaz gráfica Tkinter con matplotlib embebido.
# Permite: entrenar, cargar audio, visualizar FFT/espectrograma y predecir instrumento.

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from dataclasses import dataclass

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

try:
    import sounddevice as sd  # Reproducción de audio
    _sd_error = None
except ImportError:  # pragma: no cover - disponible condicionalmente
    sd = None
    _sd_error = "La librería 'sounddevice' no está instalada en este entorno."
except Exception as exc:  # pragma: no cover - disponible condicionalmente
    sd = None
    _sd_error = str(exc)

from .train_model import train_and_save
from .features import load_audio
from .predict import predict_file
from . import visualize

# Ruta del modelo (carpeta superior a src/)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trained_model.pkl")

@dataclass
class AppState:
    data_dir: str = ""
    audio_path: str = ""
    y = None
    sr = None
    is_playing: bool = False

class InstrumentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Instrument FFT GUI")
        self.state = AppState()

        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(main, width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        sidebar.pack_propagate(False)

        plot_area = tk.Frame(main)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        actions = tk.LabelFrame(sidebar, text="Acciones")
        actions.pack(fill=tk.X)

        self.btn_select_data = tk.Button(actions, text="Seleccionar datos", command=self.select_data_dir)
        self.btn_select_data.pack(fill=tk.X, padx=6, pady=(6, 2))

        self.btn_train = tk.Button(actions, text="Entrenar modelo", command=self.train_model)
        self.btn_train.pack(fill=tk.X, padx=6, pady=2)

        self.btn_load_audio = tk.Button(actions, text="Cargar audio (.wav)", command=self.load_audio_file)
        self.btn_load_audio.pack(fill=tk.X, padx=6, pady=2)

        self.btn_play = tk.Button(actions, text="Reproducir audio", command=self.toggle_playback)
        self.btn_play.pack(fill=tk.X, padx=6, pady=2)

        self.btn_predict = tk.Button(actions, text="Clasificar audio", command=self.predict_current)
        self.btn_predict.pack(fill=tk.X, padx=6, pady=(2, 6))

        self.status = tk.StringVar(value="Listo.")
        status_label = tk.Label(sidebar, textvariable=self.status, anchor="w", wraplength=240)
        status_label.pack(fill=tk.X, pady=(6, 4))

        self.summary_text = tk.StringVar(value="Carga un audio para ver métricas clave.")
        summary_frame = tk.LabelFrame(sidebar, text="Resumen espectral")
        summary_frame.pack(fill=tk.X, pady=4)
        tk.Label(summary_frame, textvariable=self.summary_text, justify=tk.LEFT, anchor="w", wraplength=240).pack(fill=tk.X, padx=6, pady=6)

        self.classification_text = tk.StringVar(value="Clasifica un audio para ver la confianza del modelo.")
        class_frame = tk.LabelFrame(sidebar, text="Clasificación")
        class_frame.pack(fill=tk.X, pady=4)
        tk.Label(class_frame, textvariable=self.classification_text, justify=tk.LEFT, anchor="w", wraplength=240).pack(fill=tk.X, padx=6, pady=6)

        notebook = ttk.Notebook(sidebar)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        guide_tab = tk.Frame(notebook)
        notebook.add(guide_tab, text="Guía")
        guide_msg = (
            "• Forma de onda: muestra el audio en el tiempo.\n"
            "• FFT magnitud: Transformada de Fourier; picos = frecuencias dominantes.\n"
            "• FFT fase: indica cómo se sincronizan esas frecuencias.\n"
            "• Espectrograma: energías de la FFT ventana a ventana."
        )
        tk.Label(guide_tab, text=guide_msg, justify=tk.LEFT, anchor="nw", wraplength=220).pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        notes_tab = tk.Frame(notebook)
        notebook.add(notes_tab, text="Pasos")
        steps_msg = (
            "1. Carga o entrena un modelo.\n"
            "2. Carga un audio y observa las gráficas.\n"
            "3. Usa 'Clasificar audio' para ver la etiqueta y confianza."
        )
        tk.Label(notes_tab, text=steps_msg, justify=tk.LEFT, anchor="nw", wraplength=220).pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # --- Figura matplotlib (4 subplots reorganizados) ---
        self.fig = Figure(figsize=(9.5, 6.5), dpi=100)
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.25)
        self.ax_wave = self.fig.add_subplot(gs[0, :])
        self.ax_fft = self.fig.add_subplot(gs[1, 0])
        self.ax_phase = self.fig.add_subplot(gs[1, 1])
        self.ax_spec = self.fig.add_subplot(gs[2, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Acciones ---
    def select_data_dir(self):
        d = filedialog.askdirectory(title="Selecciona la carpeta raíz de datos (contiene subcarpetas por instrumento)")
        if d:
            # 👇 Si eliges la raíz del proyecto, automáticamente usa /data
            if os.path.isdir(os.path.join(d, "data")):
                d = os.path.join(d, "data")
            self.state.data_dir = d
            self.status.set(f"Carpeta de datos: {d}")
            print("DEBUG: Carpeta de datos seleccionada =", d)

    def train_model(self):
        if not self.state.data_dir:
            messagebox.showwarning("Falta carpeta", "Primero selecciona la carpeta raíz de datos.")
            return
        try:
            info = train_and_save(self.state.data_dir, MODEL_PATH, model_type="knn", n_neighbors=5)
            acc_msg = f"{info['val_accuracy']:.3f}" if info['val_accuracy'] >= 0 else "N/A"
            msg = f"Modelo entrenado correctamente con {info['num_samples']} muestras.\nClases: {', '.join(info['classes'])}\nAcc. val: {acc_msg}"
            self.status.set(msg)
            messagebox.showinfo("Entrenamiento", msg)
        except Exception as e:
            messagebox.showerror("Error de entrenamiento", str(e))

    def load_audio_file(self):
        p = filedialog.askopenfilename(title="Selecciona un archivo .wav", filetypes=[("WAV", "*.wav *.wave")])
        if p:
            try:
                y, sr = load_audio(p)
                self.stop_playback()
                self.state.audio_path = p
                self.state.y, self.state.sr = y, sr
                self.status.set(f"Audio cargado: {os.path.basename(p)} (sr={sr})")
                self.update_plots()
            except Exception as e:
                messagebox.showerror("Error al cargar audio", str(e))

    def toggle_playback(self):
        if self.state.y is None or self.state.sr is None:
            messagebox.showwarning("Sin audio", "Carga primero un archivo .wav para reproducir.")
            return
        if sd is None:
            detail = _sd_error or "Instala 'sounddevice' para habilitar la reproducción: pip install sounddevice"
            if "PortAudio" in detail or "portaudio" in detail.lower():
                detail += "\nEn Linux instala además la librería del sistema: sudo apt-get install libportaudio2"
            messagebox.showwarning("Dependencia faltante", detail)
            return

        if self.state.is_playing:
            self.stop_playback()
            self.status.set("Reproducción detenida.")
        else:
            self.start_playback()

    def start_playback(self):
        if sd is None or self.state.y is None:
            return
        try:
            sd.stop()
            sd.play(self.state.y, samplerate=self.state.sr, blocking=False)
            self.state.is_playing = True
            self.btn_play.configure(text="Detener audio")
            self.status.set(f"Reproduciendo: {os.path.basename(self.state.audio_path)}")
            # Revisa el estado periódicamente para actualizar la UI cuando finalice.
            self.after(200, self.poll_playback)
        except Exception as e:
            messagebox.showerror("Error de reproducción", str(e))

    def stop_playback(self):
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        self.state.is_playing = False
        self.btn_play.configure(text="Reproducir audio")

    def poll_playback(self):
        if not self.state.is_playing:
            return
        if sd is None:
            return
        try:
            stream = sd.get_stream()
            is_active = bool(stream and stream.active)
        except Exception:
            is_active = False
        if not is_active:
            self.stop_playback()
            self.status.set("Reproducción finalizada.")
        else:
            self.after(200, self.poll_playback)

    def on_close(self):
        self.stop_playback()
        self.destroy()

    def update_plots(self):
        if self.state.y is None:
            return
        visualize.waveform(self.state.y, self.state.sr, self.ax_wave)
        visualize.spectrum_fft(self.state.y, self.state.sr, self.ax_fft)
        visualize.spectrum_phase(self.state.y, self.state.sr, self.ax_phase)
        visualize.spectrogram(self.state.y, self.state.sr, self.ax_spec)
        self.fig.tight_layout(rect=(0, 0, 0.97, 1.0))
        self.canvas.draw()
        self.update_summary()

    def update_summary(self):
        if self.state.y is None:
            return
        info = visualize.analyze_audio(self.state.y, self.state.sr)
        dominant = info.get("dominant_freqs", [])
        dom_notes = info.get("dominant_notes", [])
        dom_pairs = []
        for freq, note in zip(dominant, dom_notes):
            if freq > 0:
                dom_pairs.append(f"{freq:.0f} Hz ({note})")
        dom_str = ", ".join(dom_pairs) if dom_pairs else "No disponible"
        summary_lines = [
            f"Duración total: {info['duration']:.2f} s",
            f"Nivel RMS (energía promedio): {info['rms']:.3f}",
            f"Frecuencias dominantes: {dom_str}",
            f"Centro espectral: {info['spectral_centroid']:.0f} Hz",
            f"Ancho de banda espectral: {info['spectral_bandwidth']:.0f} Hz",
        ]
        self.summary_text.set("\n".join(summary_lines))

    def predict_current(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning("Modelo no encontrado", "Primero entrena y guarda el modelo.")
            return
        if not self.state.audio_path:
            messagebox.showwarning("Sin audio", "Carga primero un archivo .wav para clasificar.")
            return
        try:
            result = predict_file(self.state.audio_path, MODEL_PATH)
            label = result.get("label", "?")
            confidence = result.get("confidence")
            top_classes = result.get("top_classes")

            if top_classes:
                header = f"Predicción: {label}"
                if confidence is not None:
                    header += f" ({confidence*100:.1f}% de confianza)"
                lines = [header, "Probabilidades principales:"]
                for cls, prob in top_classes:
                    marker = "→" if cls == label else "•"
                    lines.append(f"   {marker} {cls}: {prob*100:.1f}%")
                class_msg = "\n".join(lines)
            else:
                proxy = result.get("neigh_weights")
                if proxy:
                    approx = max(proxy) * 100
                    class_msg = f"→ {label} (peso relativo {approx:.1f}%)"
                else:
                    class_msg = f"→ {label}"

            self.classification_text.set(class_msg)
            self.status.set(f"Instrumento predicho: {label}")
            messagebox.showinfo("Predicción", class_msg)
        except Exception as e:
            messagebox.showerror("Error en predicción", str(e))

def run():
    app = InstrumentApp()
    app.mainloop()
