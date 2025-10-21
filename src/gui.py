# Interfaz gráfica Tkinter con matplotlib embebido.
# Permite: entrenar, cargar audio, visualizar FFT/espectrograma y predecir instrumento.

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

class InstrumentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Instrument FFT GUI")
        self.state = AppState()

        # --- Controles superiores ---
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        self.btn_select_data = tk.Button(top, text="Seleccionar carpeta de datos", command=self.select_data_dir)
        self.btn_select_data.pack(side=tk.LEFT, padx=4)

        self.btn_train = tk.Button(top, text="Entrenar modelo", command=self.train_model)
        self.btn_train.pack(side=tk.LEFT, padx=4)

        self.btn_load_audio = tk.Button(top, text="Cargar audio (.wav)", command=self.load_audio_file)
        self.btn_load_audio.pack(side=tk.LEFT, padx=4)

        self.btn_predict = tk.Button(top, text="Clasificar audio", command=self.predict_current)
        self.btn_predict.pack(side=tk.LEFT, padx=4)

        # --- Estado ---
        self.status = tk.StringVar(value="Listo.")
        tk.Label(self, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=8)

        # --- Figura matplotlib (3 subplots) ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_wave = self.fig.add_subplot(3,1,1)
        self.ax_fft  = self.fig.add_subplot(3,1,2)
        self.ax_spec = self.fig.add_subplot(3,1,3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
                self.state.audio_path = p
                self.state.y, self.state.sr = y, sr
                self.status.set(f"Audio cargado: {os.path.basename(p)} (sr={sr})")
                self.update_plots()
            except Exception as e:
                messagebox.showerror("Error al cargar audio", str(e))

    def update_plots(self):
        if self.state.y is None:
            return
        visualize.waveform(self.state.y, self.state.sr, self.ax_wave)
        visualize.spectrum_fft(self.state.y, self.state.sr, self.ax_fft)
        visualize.spectrogram(self.state.y, self.state.sr, self.ax_spec)
        self.fig.tight_layout()
        self.canvas.draw()

    def predict_current(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning("Modelo no encontrado", "Primero entrena y guarda el modelo.")
            return
        if not self.state.audio_path:
            messagebox.showwarning("Sin audio", "Carga primero un archivo .wav para clasificar.")
            return
        try:
            pred = predict_file(self.state.audio_path, MODEL_PATH)
            self.status.set(f"Instrumento predicho: {pred}")
            messagebox.showinfo("Predicción", f"Instrumento predicho: {pred}")
        except Exception as e:
            messagebox.showerror("Error en predicción", str(e))

def run():
    app = InstrumentApp()
    app.mainloop()