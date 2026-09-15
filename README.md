[![CI](https://github.com/AgusGall0/swim-stroke-analyzer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AgusGall0/swim-stroke-analyzer/actions/workflows/ci.yml)

# swim-stroke-analyzer

Análisis biomecánico de la brazada de crol a partir de video, con visión
computacional.

> **Estado: andamiaje.** Hoy el proyecto no analiza video. Tiene la estructura
> del paquete, la configuración validada, la descarga verificada del modelo y un
> CLI que valida argumentos. La extracción de landmarks y todo lo que sigue
> todavía no están implementados.

## Qué es y cuál es el objetivo

El objetivo es tomar video de un nadador de crol y producir cinemática
cuantitativa de la brazada: ángulos articulares por fotograma, ciclos de
brazada segmentados y métricas derivadas, con una caracterización de cuánto
error tienen esas mediciones.

Correr un modelo de pose sobre un video es la parte fácil. El trabajo del
proyecto está en lo que viene después: persistir los datos, procesar la señal,
calcular métricas con criterio biomecánico y validarlas contra anotación
manual.

El primer entregable previsto es un camino completo y angosto:

```
video → landmarks a Parquet → filtrado → ángulos bilaterales
      → segmentación de ciclos → curva media normalizada al 100% del ciclo
```

que termina en la figura de la curva media ± desvío estándar del ángulo de codo
a lo largo del ciclo de brazada.

Es un proyecto de Juan Agustín Gallo (Ingeniería en Informática, Facultad de
Tecnología y Ciencias Aplicadas, U.N.Ca.), en etapa piloto.

## Qué hace hoy

- **Paquete `swimalyzer`** instalable, con los módulos `io`, `pose`, `signal`,
  `metrics` y `viz` creados pero vacíos: cada uno solo documenta qué va a
  contener.
- **`config.yaml` con validación.** Si falta un parámetro, sobra uno
  desconocido o un valor está fuera de rango, la carga falla nombrando el
  parámetro.
- **Descarga del modelo** MediaPipe Pose Landmarker (heavy) con verificación de
  SHA-256.
- **CLI** `swimalyzer extraer`, que valida el video, el directorio de salida, la
  configuración y el recorte, y termina avisando que la extracción no está
  implementada.
- **Tests y CI** (lint con ruff y pytest en Python 3.11).

No hay extracción de landmarks, filtrado, cálculo de ángulos, segmentación de
ciclos ni figuras. No hay análisis en tiempo real ni soporte para otros estilos.

## Instalación

Requiere **Python 3.11 o superior** (lo exigen las versiones pinneadas de
numpy, scipy y pandas).

```bash
git clone https://github.com/AgusGall0/swim-stroke-analyzer.git
cd swim-stroke-analyzer

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # sin [dev] no instala pytest ni ruff
```

En Debian/Ubuntu, si `python3 -m venv` falla porque falta `ensurepip`, instalá
el paquete del sistema correspondiente (por ejemplo `sudo apt install
python3.14-venv`, según tu versión de Python).

Las dependencias están pinneadas en `pyproject.toml`. OpenCV se instala como
`opencv-contrib-python`, que es el que pide MediaPipe: no instales además
`opencv-python`, porque los dos escriben el mismo módulo `cv2`.

### Modelo

El modelo no se versiona. Para descargarlo:

```bash
python scripts/descargar_modelo.py
```

Descarga `pose_landmarker_heavy.task` desde el bucket de modelos de MediaPipe,
verifica su SHA-256 contra el valor de `config.yaml` y lo deja en `modelos/`
(ignorada por git). Si el hash no coincide, descarta el archivo y termina con
error. Si el modelo ya está y es correcto, no descarga nada.

## Uso

```bash
swimalyzer --help
swimalyzer extraer <video> --out <directorio> [--config config.yaml] [--recorte X Y ANCHO ALTO]
```

Por ahora `extraer` solo valida los argumentos y termina con código de salida 1
avisando que no está implementado. `--recorte` recibe la región del video
original a procesar, en píxeles, y sobrescribe `video.recorte` de la
configuración.

### Configuración

Todos los parámetros viven en `config.yaml`, cada uno con un comentario que
explica qué controla. Los que corresponden a decisiones metodológicas todavía
abiertas están en `null` a propósito, sin un valor provisorio:

- umbral de `visibility` para descartar muestras,
- tipo, orden y frecuencia de corte del filtro,
- criterio de segmentación de ciclos,
- método de corrección de intercambios izquierda/derecha.

Cada uno se va a fijar con evidencia sobre datos reales. El código que los use
los pide con `Configuracion.exigir(...)`, que falla mientras sigan en `null`.

### Tests

```bash
pytest
ruff check . && ruff format --check .
```

## Estructura

```
src/swimalyzer/
  cli.py       subcomandos de línea de comandos
  config.py    carga y validación de config.yaml
  io/          lectura de video, escritura de Parquet, metadata (vacío)
  pose/        índices de landmarks; wrapper de MediaPipe (pendiente)
  signal/      interpolación, filtrado, suavizado (vacío)
  metrics/     ángulos, segmentación de ciclos, métricas derivadas (vacío)
  viz/         generación de figuras (vacío)
tests/
config.yaml
scripts/       descarga del modelo
experiments/   scripts originales, conservados como registro del proceso
```

`experiments/` contiene los primeros prototipos (OpenCV con webcam, prueba de
instalación de MediaPipe, y un script que dibuja el esqueleto y el ángulo del
codo derecho en pantalla). No se usan ni se mantienen; su README explica qué
era cada uno.

## Estado y hoja de ruta

**Hecho (Fase 0, andamiaje):**

- [x] Estructura de paquete y `pyproject.toml` con versiones pinneadas
- [x] `config.yaml` con validación
- [x] Script de descarga del modelo con verificación de hash
- [x] Esqueleto del CLI
- [x] Tests, CI y licencia
- [x] Prototipos originales movidos a `experiments/`

**Falta (rebanada vertical):**

- [ ] Extracción de landmarks a Parquet, una fila por landmark por fotograma,
      registrando los fotogramas sin detección
- [ ] Metadata de cada corrida (fps, resolución, hash del modelo, versión del
      código, parámetros)
- [ ] Distribución de `visibility` por landmark y tasa de descarte
- [ ] Interpolación y filtrado de las trayectorias
- [ ] Ángulos articulares bilaterales en coordenadas de píxel
- [ ] Detección de intercambios izquierda/derecha
- [ ] Segmentación de ciclos de brazada
- [ ] Figura de curva media ± desvío estándar del ángulo de codo

**Después:** con esa rebanada funcionando se decide el alcance real según qué
tan bien funcione MediaPipe en estas condiciones, y se valida contra anotación
manual.

## Limitaciones conocidas del enfoque

Estas limitaciones son del método, no bugs, y parte del trabajo es medir cuánto
afectan:

- **Modelo entrenado fuera del agua.** MediaPipe Pose no fue entrenado para
  nadadores sumergidos. Posturas horizontales, cuerpo parcialmente sumergido y
  el aspecto del cuerpo bajo el agua están lejos de sus datos de
  entrenamiento.
- **Refracción.** La luz se desvía al pasar entre agua, vidrio y aire.
  Distorsiona la geometría aparente del cuerpo, sobre todo cerca de los bordes
  de la ventana de observación y cuando la cámara no está perpendicular a ella.
- **Burbujas y turbulencia.** La entrada de la mano y la patada generan
  burbujas que ocultan justo los segmentos distales (manos, pies) que más
  interesan.
- **Línea de superficie.** Con la cámara a la altura del agua, el cuerpo queda
  partido entre una parte aérea y una sumergida, con reflejos en la cara
  inferior de la superficie. En el recobro el brazo sale del agua y cambia de
  medio.
- **Oclusión e intercambio de lados.** En vista lateral, el brazo más alejado
  queda tapado por el torso durante parte del ciclo, y la rotación del cuerpo
  puede hacer que el modelo confunda izquierda y derecha.
- **Solo 2D.** Los ángulos se miden en el plano de la imagen. La profundidad `z`
  que estima MediaPipe no se usa: es la componente menos confiable y bajo el
  agua no tiene sentido. Un ángulo proyectado difiere del ángulo real cuando el
  segmento no es paralelo al plano de la cámara.
- **Resolución.** En el video de desarrollo el nadador ocupa pocos píxeles, así
  que los landmarks distales son ruidosos.

Ningún resultado de este proyecto está validado todavía contra anotación
manual.

## Datos

El video de desarrollo es de terceros y no se versiona, ni el archivo ni
fotogramas extraídos. Tampoco se versionan el modelo ni los datos generados.
El material definitivo se va a grabar con consentimiento de los nadadores.

## Licencia

[MIT](LICENSE) © 2026 Juan Agustín Gallo
