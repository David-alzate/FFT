# instrument_fft_gui

Aplicación educativa en Python para analizar audio instrumental, explicar la Transformada de Fourier desde el punto de vista matemático y práctico, y clasificar instrumentos mediante modelos clásicos de *machine learning*.

## Tabla de contenido
1. [Tecnologías y requisitos](#tecnologías-y-requisitos)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Instalación y ejecución](#instalación-y-ejecución)
4. [Recorrido por la interfaz](#recorrido-por-la-interfaz)
5. [Visualizaciones: interpretación técnica y cotidiana](#visualizaciones-interpretación-técnica-y-cotidiana)
6. [Panel «Resumen espectral»](#panel-resumen-espectral)
7. [Clasificación de instrumentos](#clasificación-de-instrumentos)
8. [Preguntas frecuentes](#preguntas-frecuentes)

## Tecnologías y requisitos
- Python 3.10+
- Librerías principales: `librosa`, `numpy`, `matplotlib`, `scikit-learn`, `pandas`, `soundfile`, `sounddevice`, `tkinter` (incluido en la librería estándar de Python)
- Sistema operativo con soporte para Tkinter y PortAudio (para reproducir audio mediante `sounddevice`)

## Estructura del proyecto
```
instrument_fft_gui/
├── data/                  # Audios de entrenamiento por instrumento (subcarpetas)
├── models/
│   └── trained_model.pkl  # Modelo entrenado (se crea tras el entrenamiento)
├── src/
│   ├── features.py        # Extracción de características
│   ├── train_model.py     # Lógica de entrenamiento y guardado
│   ├── predict.py         # Carga del modelo y predicción
│   ├── visualize.py       # Gráficos (forma de onda, FFT, STFT)
│   └── gui.py             # Interfaz Tkinter
├── main.py                # Punto de entrada (ejecuta la GUI)
├── requirements.txt
├── README.md
└── setup.sh
```

> El directorio `data/` puede contener subcarpetas como `piano/`, `guitar/`, `violin/`. Cada subcarpeta reúne archivos `.wav` mono del instrumento correspondiente.

## Instalación y ejecución
1. **Crear entorno virtual (opcional pero recomendado)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Linux) Instalar soporte para Tk/PortAudio** si fuese necesario:
   ```bash
   sudo apt-get install python3-tk libportaudio2
   ```

4. **Ejecutar la aplicación**:
   ```bash
   python main.py
   ```

## Recorrido por la interfaz
La ventana se divide en dos grandes áreas:

### Panel lateral (izquierda)
- **Acciones**: botones para seleccionar la carpeta de datos, entrenar el modelo, cargar un audio, reproducirlo y clasificarlo.
- **Estado**: mensajes cortos sobre la última acción ejecutada.
- **Resumen espectral**: estadísticas del audio cargado (explicadas en detalle más adelante).
- **Clasificación**: muestra la predicción actual, la confianza estimada y las probabilidades principales.
- **Pestañas de ayuda**:
  - *Guía*: recordatorio rápido de qué representa cada gráfica.
  - *Pasos*: orden sugerido para usar la aplicación.

### Área de gráficas (derecha)
Se compone de cuatro subgráficas:
1. Forma de onda (tiempo)
2. Transformada de Fourier - magnitud
3. Transformada de Fourier - fase
4. Espectrograma (STFT)

Cada panel se actualiza al cargar un audio `.wav` y permite ilustrar la teoría de Fourier con ejemplos reales.

## Visualizaciones: interpretación técnica y cotidiana
A continuación se describe cada gráfica desde dos perspectivas: la matemática (para sustentar un trabajo académico) y la cotidiana (para comunicar a cualquier audiencia).

### 1. Forma de onda (dominio del tiempo)
- **Matemática**: grafica la función de señal `x(t)` en amplitud versus tiempo. Puede interpretarse como la solución experimental de la ecuación de onda `u_tt = c^2 u_xx`, donde el instrumento fija las condiciones iniciales y de frontera. Permite observar transitorios, silencios y envolventes.
- **Cotidiana**: muestra cómo vibra el instrumento a lo largo del tiempo. Los picos altos indican golpes o ataques; los segmentos casi planos son silencios o decaimiento.

### 2. Transformada de Fourier - magnitud (log-dB)
- **Matemática**:
  - Calcula una FFT unilateral aproximando la integral de Fourier `X(f) = ∫ x(t) e^{-j 2π f t} dt` mediante `numpy.fft.rfft`.
  - Se representa `20 * log10 |X(f)|` en escala logarítmica de frecuencia para abarcar componentes desde graves hasta agudos.
  - Las líneas rojas verticales señalan los tres picos de mayor energía (p1, p2, p3). En la esquina superior derecha aparece una tarjeta con la frecuencia exacta y la nota musical aproximada (con afinación A4 = 440 Hz). Estos picos corresponden a la frecuencia fundamental y a los primeros armónicos o modos propios de la expansión en Fourier.
- **Cotidiana**:
  - Muestra de qué frecuencias está hecha la nota. La barra roja `p1` indica «la altura principal» del sonido; `p2` y `p3` son los armónicos que le dan color o timbre. Si se mueven, cambia la percepción del instrumento.

### 3. Transformada de Fourier - fase
- **Matemática**: grafica el ángulo de `X(f)` en grados. La fase codifica el desplazamiento temporal de cada componente sinusoidal. Aunque la magnitud determina «qué tanto» de cada frecuencia hay, la fase indica «en qué instante» se combinan para reconstruir la señal.
- **Cotidiana**: ayuda a explicar que para recrear el audio original no basta con conocer la potencia de cada frecuencia; también hay que saber cómo se alinean en el tiempo. Cambiar la fase altera el timbre o produce cancelaciones.

### 4. Espectrograma (STFT)
- **Matemática**: aplica la Transformada de Fourier a ventanas cortas (Short-Time Fourier Transform). Cada columna es una FFT de `x(t)` multiplicada por una ventana deslizante. Se representa en dB con `librosa.amplitude_to_db`, y se etiquetan ejes en segundos (horizontal) y Hertz (vertical). Permite analizar señales no estacionarias, como ataques, vibratos o glisandos.
- **Cotidiana**: es «un mapa de calor» que muestra cuánta energía hay en cada frecuencia mientras la nota suena. Los colores brillantes representan partes del espectro donde el instrumento tiene más fuerza en ese instante.

## Panel «Resumen espectral»
| Métrica | Descripción matemática | Interpretación cotidiana |
| --- | --- | --- |
| **Duración total** | Se calcula como `N / fs`, con `N` muestras y frecuencia de muestreo `fs`. | Cuánto dura el audio cargado. |
| **Nivel RMS** | `sqrt(sum(x[n]^2) / N)`. Mide la energía promedio de la señal. | Qué tan fuerte o suave es el sonido en promedio. |
| **Frecuencias dominantes** | Se extraen los picos principales de la magnitud FFT. Se muestra frecuencia y nota equivalente. | «Notas» que están predominando. Sirve para ver si la nota está afinada. |
| **Centro espectral** | Promedio ponderado de las frecuencias con sus magnitudes: `sum(fk * |X(fk)|) / sum(|X(fk)|)`. | Sensación de brillo del sonido: valores altos -> sonido más brillante, valores bajos -> sonido más grave. |
| **Ancho de banda** | Desviación estándar alrededor del centro espectral. | Qué tan dispersa está la energía en las frecuencias. |

> Las frecuencias y notas dominantes coinciden con la tarjeta roja de la gráfica de magnitud; un pico alto en la gráfica se refleja aquí.

## Clasificación de instrumentos
La app aprende a diferenciar instrumentos a partir de características espectrales.

### 1. Extracción de características
Archivo: `src/features.py`
- **MFCC (Mel-frequency cepstral coefficients)**: describen la envolvente espectral percibida; se calculan 13 coeficientes y se agregan media y desviación estándar.
- **Centroid, bandwidth y rolloff**: métricas clásicas del espectro (se registran media, desviación, mínimo y máximo). 
- Las características forman un vector numérico (aprox. 44 valores) por audio.

### 2. Entrenamiento del modelo
Archivo: `src/train_model.py`
- Recorre las subcarpetas de `data/` y construye un dataset etiquetado (`instrumento` = nombre de la carpeta).
- Divide en entrenamiento/validación (80/20) para estimar la exactitud.
- Permite dos modelos clásicos:
  - **K-Nearest Neighbors (KNN)**: clasifica según los `k` audios más parecidos. Matemáticamente, toma los vectores de características y calcula distancias Euclidianas; la clase más frecuente entre los vecinos define la etiqueta.
  - **Random Forest (RF)**: ensamble de árboles de decisión; cada árbol vota por un instrumento, y la decisión final es el promedio de votos (probabilidad).
- El resultado se guarda en `models/trained_model.pkl` para reutilizarse.

### 3. Predicción y probabilidades
Archivo: `src/predict.py`
- Al clasificar un nuevo audio, se replica el proceso de extracción de características y se ordenan siguiendo las columnas usadas en el entrenamiento.
- **Probabilidades**:
  - Si el modelo implementa `predict_proba` (KNN o RF en scikit-learn), obtienes las probabilidades `P(clase | audio)`. En KNN equivalen al porcentaje de vecinos que pertenecen a cada clase; en RF al promedio de probabilidades de todos los árboles.
  - Se ordenan de mayor a menor y se muestran los tres valores superiores.
  - La «confianza» que ves es la probabilidad de la clase ganadora.
- Si el modelo no ofrece `predict_proba`, se cae a un proxy con los pesos de los vecinos más cercanos; aun así se muestra la etiqueta ganadora.

### 4. Interpretación no técnica
- El modelo compara el «perfil espectral» del audio con los ejemplos almacenados. Si suena parecido al conjunto de audios etiquetados como `violin`, su probabilidad aumenta para esa clase.
- Los porcentajes responden a la pregunta: «¿Qué tan similares fueron los patrones del audio nuevo a cada instrumento conocido?».

## Preguntas frecuentes
**¿Qué tan grande debe ser el dataset?**
> Lo ideal es tener decenas de ejemplos por instrumento, cubriendo diferentes notas y articulaciones. El modelo KNN necesita variabilidad para no confundir instrumentos muy parecidos.

**¿Se puede cambiar el modelo?**
> Sí. `train_model.py` admite `model_type="knn"` o `"rf"`. Puedes extenderlo con otros modelos de scikit-learn ajustando el guardado en `train_and_save`.

**¿Por qué la FFT está en dB y eje logarítmico?**
> El oído humano percibe diferencias relativas (logarítmicas). Usar dB y eje log permite observar armónicos débiles que serían invisibles en escala lineal.

**¿Qué representa la tarjeta roja con p1/p2/p3?**
> Resume las tres frecuencias más energéticas. `p1` ≈ frecuencia fundamental (lo que percibimos como altura). `p2` y `p3` suelen ser armónicos (múltiplos enteros o casi enteros) que definen el color del instrumento.
