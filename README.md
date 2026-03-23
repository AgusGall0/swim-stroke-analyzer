# Analizador de Brazada de Crol con Visión Computacional

Este proyecto utiliza visión computacional y modelos de aprendizaje profundo para analizar la técnica de natación (estilo crol) a partir de archivos de video. El sistema identifica y rastrea los puntos de referencia articulares del nadador para permitir un futuro análisis biomecánico (cálculo de ángulos de codo, hombro y entrada al agua).

## Estado Actual del Proyecto
El desarrollo se encuentra en la fase de extracción y visualización de datos cinemáticos.
* **Lectura de video:** Implementada mediante OpenCV.
* **Inferencia de IA:** Integración completa con la Tasks API de MediaPipe.
* **Extracción de coordenadas:** Detección de 33 puntos de referencia anatómicos (Pose Landmarks) por cada frame válido.
* **Renderizado:** Dibujo del esqueleto y conexiones sobre el video original en tiempo real.

## Arquitectura y Tecnologías
* **Lenguaje:** Python
* **Procesamiento de Imágenes:** OpenCV (`cv2`)
* **Motor de Inteligencia Artificial:** MediaPipe (`mediapipe`)
* **Modelo de Detección:** Pose Landmarker (Versión: Heavy)

## Requisitos del Sistema e Instalación

### 1. Entorno de Ejecución
Es estrictamente necesario utilizar **Python 3.9 o superior** (el proyecto fue probado y configurado en Python 3.13). Versiones anteriores (como Python 3.8) generarán errores fatales de sintaxis (`TypeError: 'type' object is not subscriptable`) debido a las anotaciones de tipado en el código fuente de MediaPipe.

### 2. Archivos del Modelo
El modelo de pesos de la red neuronal no está incluido en el código fuente. Debes descargar el modelo preentrenado de Google:
* Descargar `pose_landmarker_heavy.task`.
* Ubicar el archivo `.task` en el directorio raíz del proyecto, al mismo nivel que los scripts de ejecución.

### 3. Configuración del Entorno Virtual
Se recomienda aislar las dependencias del proyecto utilizando un entorno virtual. Ejecutar los siguientes comandos en la terminal:

```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual (Windows)
.\venv\Scripts\activate

# Instalar las librerías requeridas
pip install opencv-python mediapipe

Estructura del Directorio
Para que el script funcione correctamente, el directorio de trabajo debe contener la siguiente estructura base:
/
├── venv/                           # Entorno virtual
├── apertura_v.py                   # Script principal de ejecución
├── pose_landmarker_heavy.task      # Pesos del modelo de MediaPipe
└── ejemplo.mp4                     # Video de muestra a analizar
Próximos Pasos (En desarrollo)
Aislamiento de las coordenadas (x, y) específicas de las extremidades superiores (hombros, codos y muñecas).

Implementación de fórmulas trigonométricas para calcular la amplitud articular durante la fase de agarre y tirón.


