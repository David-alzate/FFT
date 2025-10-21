# instrument_fft_gui

Proyecto educativo en Python para **reconocer el instrumento** (piano, guitarra, violín, etc.) que emite una **nota musical** utilizando **análisis de Fourier (FFT)** y **características espectrales** clásicas, **sin redes neuronales**. Incluye una **interfaz gráfica con Tkinter** para entrenar, clasificar y visualizar forma de onda, espectro (FFT) y espectrograma.

## 🧰 Tecnologías
- Python 3.10+
- Librerías: `librosa`, `numpy`, `matplotlib`, `scikit-learn`, `pandas`, `tkinter` (estándar), `soundfile` (backend para `librosa`)

## 📦 Estructura
```
instrument_fft_gui/
├── data/
│   ├── piano/
│   ├── guitar/
│   ├── violin/
│   └── test/
├── models/
│   └── trained_model.pkl
├── src/
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│   ├── visualize.py
│   └── gui.py
├── main.py
├── requirements.txt
├── README.md
└── setup.sh
```

> Se incluyen **pequeños audios de ejemplo** (seno de 1s) en `data/piano|guitar|violin|test` para ilustrar la estructura.

## 🚀 Instalación
1. (Opcional) Crear entorno virtual e instalar dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Verifica que tienes Tk instalado (en Linux puede requerir `sudo apt-get install python3-tk`).

## 🎛️ Uso
1. Coloca tus archivos `.wav` en subcarpetas dentro de `data/`, una por instrumento, por ejemplo:
   ```
   data/piano/mi_archivo.wav
   data/guitar/otro.wav
   data/violin/notaX.wav
   ```
2. Ejecuta la app:
   ```bash
   python main.py
   ```
3. En la interfaz:
   - **Entrenar modelo**: selecciona la carpeta raíz `data/`. El sistema recorre subcarpetas (una por instrumento), extrae features (`centroid`, `bandwidth`, `rolloff`, `mfcc`) y entrena un `KNeighborsClassifier` (o `RandomForest`). Se guarda automáticamente en `models/trained_model.pkl`.
   - **Clasificar audio**: selecciona un `.wav` para predecir el instrumento con el modelo entrenado.
   - **Visualizar FFT / Espectrograma**: al cargar un audio, se muestran forma de onda, espectro (FFT con `numpy.fft.fft`) y espectrograma logarítmico.

## 🧠 Notas técnicas
- Lectura de audios con `librosa.load()` (mono, SR por defecto 22050 Hz).
- FFT con `numpy.fft.fft` y magnitud `|X[k]|`.
- Espectrograma con `librosa.stft` y escala en dB (`librosa.amplitude_to_db`).
- Features espectrales con `librosa.feature.*` + agregaciones (media, desviación estándar).
- Modelo con `scikit-learn` y persistencia con `pickle`.

## 📈 Consejos de dataset
- Idealmente usa **varias notas por instrumento** y diferentes articulaciones/ataques.
- Normaliza volúmenes y evita ruido de fondo excesivo.
