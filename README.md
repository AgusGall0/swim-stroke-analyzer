[![CI](https://github.com/AgusGall0/swim-stroke-analyzer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AgusGall0/swim-stroke-analyzer/actions/workflows/ci.yml)

# swim-stroke-analyzer

Análisis biomecánico de la brazada de crol a partir de video, con visión
computacional.

> **Estado: media rebanada vertical.** Del video salen landmarks persistidos en
> Parquet, un informe que caracteriza la señal, y trayectorias interpoladas y
> filtradas. Los ángulos articulares, la segmentación de ciclos y la figura
> final todavía no están implementados.

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
video → landmarks a Parquet → filtrado → ángulos articulares
      → segmentación de ciclos → curva media normalizada al 100% del ciclo
```

que termina en la figura de la curva media ± desvío estándar del ángulo de codo
a lo largo del ciclo de brazada. Los ángulos se calculan sobre el lado cercano
a la cámara; el porqué está en
[Qué se midió sobre el video de desarrollo](#qué-se-midió-sobre-el-video-de-desarrollo).

Es un proyecto de Juan Agustín Gallo (Ingeniería en Informática, Facultad de
Tecnología y Ciencias Aplicadas, U.N.Ca.), en etapa piloto.

## Qué hace hoy

- **`swimalyzer extraer`**: corre MediaPipe Pose Landmarker fotograma a
  fotograma y escribe un Parquet con una fila por landmark por fotograma
  (`frame, timestamp_ms, landmark_id, x, y, z, visibility, presence`), más un
  JSON de metadata con fps, resolución, recorte, SHA-256 del modelo y del video,
  versión del código y la configuración usada. Los fotogramas sin detección se
  escriben con `NaN` y sus índices quedan listados: no se saltean en silencio.
- **`scripts/caracterizar_senal.py`**: informe reproducible de la señal cruda —
  distribución de `visibility` por landmark, cuánto descarta cada umbral,
  contenido frecuencial de las trayectorias, análisis de residuos de Winter y
  candidatos a intercambio izquierda/derecha, con seis figuras.
- **`swimalyzer filtrar`**: interpola los huecos cortos, filtra las trayectorias
  en píxeles con un Butterworth sin desfase y marca los fotogramas sospechados
  de intercambio, sin corregirlos. No descarta nada por `visibility`.
- **`swimalyzer angulos`**: calcula los ángulos articulares del lado cercano a
  la cámara (codo, hombro y rodilla izquierdos) sobre coordenadas en píxeles.
  Cada ángulo hereda las banderas de sus tres landmarks: queda marcado si a
  alguno le faltó pasar por el filtro, se interpoló, quedó sospechado de
  intercambio o tiene `visibility` por debajo del umbral de reporte. Marcar no
  es descartar: el valor se guarda igual.
- **`scripts/informe_angulos.py`**: informe reproducible de los ángulos —
  cobertura, porcentaje marcado desglosado por motivo, rango de valores de cada
  articulación y la serie temporal del ángulo de codo.
- **`config.yaml` con validación.** Si falta un parámetro, sobra uno desconocido
  o un valor está fuera de rango, la carga falla nombrando el parámetro.
- **Descarga del modelo** MediaPipe Pose Landmarker (heavy) con verificación de
  SHA-256.
- **Tests y CI** (lint con ruff y pytest en Python 3.11).

No hay segmentación de ciclos ni figura de curva media del ciclo de brazada: el
criterio de segmentación es una decisión abierta. No hay análisis en tiempo real
ni soporte para otros estilos. Ningún resultado está validado contra anotación
manual.

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

# 1. video → landmarks + metadata
swimalyzer extraer <video> --out <directorio> [--recorte X Y ANCHO ALTO] [--fps FPS]

# 2. landmarks → informe de caracterización (figuras + números)
python scripts/caracterizar_senal.py <directorio>

# 3. landmarks → trayectorias interpoladas y filtradas
swimalyzer filtrar <directorio> [--out <otro directorio>]

# 4. trayectorias filtradas → ángulos articulares del lado cercano
swimalyzer angulos <directorio> [--out <otro directorio>]

# 5. ángulos → informe de ángulos (figuras + números)
python scripts/informe_angulos.py <directorio>
```

`--recorte` recibe la región del video original a procesar, en píxeles, y
sobrescribe `video.recorte` de la configuración. `--fps` solo hace falta si el
contenedor no informa una tasa de fotogramas usable; sin él, y sin dato en el
contenedor, el comando falla en vez de calcular timestamps con una división por
cero.

`filtrar` escribe `landmarks_filtrados.parquet` con las coordenadas ya en
píxeles (`x_px`, `y_px`), la `visibility` de cada muestra y tres banderas:
`interpolado`, `filtrado` (falso en los tramos demasiado cortos para `filtfilt`,
que quedan sin filtrar en lugar de desaparecer) e `intercambio_sospechado`.

`angulos` escribe `angulos.parquet` con una fila por articulación por fotograma:
el ángulo en grados, la peor `visibility` de los tres landmarks que lo definen y
una bandera por motivo de marcado (`sin_filtrar`, `interpolado`,
`intercambio_sospechado`, `visibility_baja`). El ángulo es el **ángulo incluido
en el vértice**, de 0° a 180°, con 180° el segmento extendido; los fotogramas
sin los tres landmarks quedan en `NaN`.

### El recorte del video de desarrollo

El clip de desarrollo es vertical, de 576×512, pero el contenido real ocupa solo
las filas 0 a 323: el resto es relleno negro con texto sobreimpreso. Para ese
video:

```bash
swimalyzer extraer crol_lateral.mp4 --out salidas/crol --recorte 0 0 576 324
```

Recortarlo sube la detección de 438 a 603 fotogramas de 1055, porque el nadador
ocupa una fracción mayor de lo que entra al modelo.

**Ese valor no va a `config.yaml`.** Un recorte es una propiedad del archivo que
se está procesando, no del método: dejarlo como valor por omisión lo aplicaría a
cualquier otro video y le comería una franja de imagen sin avisar. `video.recorte`
queda en `null` y el recorte se pasa por línea de comandos; la metadata de cada
corrida registra cuál se usó.

### Configuración

Todos los parámetros viven en `config.yaml`, cada uno con un comentario que
explica qué controla y, cuando corresponde, de dónde salió el valor.

Decisiones ya tomadas, con el informe de caracterización a la vista:

| Parámetro | Valor | Por qué |
|---|---|---|
| `calidad.umbral_visibility_reporte` | `0.3` | Criterio de reporte, no de descarte. Un umbral global no elige qué fotogramas son malos: elige qué miembros existen. |
| `filtrado.tipo` / `orden` | `butterworth` / `2` | Con `filtfilt`, sin desfase. El estándar en biomecánica. |
| `filtrado.frecuencia_corte_hz` | `3.4` | Mediana del análisis de residuos de Winter sobre este material (2.0 a 4.25 Hz según el landmark). |
| `filtrado.hueco_maximo_interpolable_fotogramas` | `3` | 0.1 s. Con la brazada a 0.47 Hz, interpolar medio segundo es inventar un cuarto de ciclo. |
| `lateralidad.metodo_correccion_intercambios` | `ninguno` | Se detectan y se marcan; corregirlos antes de ver si el artefacto llega a los ángulos sería tocar el dato sin evidencia. |

Siguen abiertas, en `null` a propósito y sin valor provisorio: el criterio de
segmentación de ciclos (`segmentacion.senal` y
`segmentacion.distancia_minima_entre_picos_s`). El código que las use las pide
con `Configuracion.exigir(...)`, que falla mientras sigan en `null`.

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
  io/          lectura de video, escritura de Parquet, metadata de corrida
  pose/        índices de landmarks, wrapper de MediaPipe, etapa de extracción
  signal/      caracterización de la señal, interpolación y filtrado
  metrics/     ángulos articulares; faltan ciclos y métricas derivadas
  viz/         paleta común y figuras de los informes
tests/
config.yaml
scripts/       descarga del modelo, informes de caracterización y de ángulos
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
- [x] CLI, tests, CI y licencia
- [x] Prototipos originales movidos a `experiments/`

**Rebanada vertical:**

- [x] Extracción de landmarks a Parquet, una fila por landmark por fotograma,
      registrando los fotogramas sin detección
- [x] Metadata de cada corrida (fps, resolución, hash del modelo y del video,
      versión del código, parámetros)
- [x] Distribución de `visibility` por landmark y tasa de descarte por umbral
- [x] Análisis frecuencial y de residuos para justificar la frecuencia de corte
- [x] Interpolación de huecos cortos y filtrado de las trayectorias
- [x] Detección y marcado de intercambios izquierda/derecha
- [x] Ángulos articulares en coordenadas de píxel, con las banderas de calidad
      heredadas de sus landmarks
- [ ] Segmentación de ciclos de brazada
- [ ] Figura de curva media ± desvío estándar del ángulo de codo

**Después:** con esa rebanada funcionando se decide el alcance real según qué
tan bien funcione MediaPipe en estas condiciones, y se valida contra anotación
manual.

## Qué se midió sobre el video de desarrollo

Los números que siguen salen de una corrida real sobre el clip de desarrollo
(576×324 tras el recorte, 30 fps, 1055 fotogramas) y están en
`caracterizacion/informe.md` de esa corrida. Son provisorios en el sentido de
que describen **este** material, no el método en general.

**Con este material el análisis bilateral no es viable.** En vista lateral el
brazo alejado de la cámara queda tapado por el torso buena parte del ciclo, y
eso se ve en los datos:

- Con umbral `0.3`, el **codo derecho** —el lado lejano— queda sin medición
  usable en el **96 % de los fotogramas** del video (93,5 % de los que tuvieron
  detección). La muñeca derecha, en el 86 %.
- La **muñeca del lado lejano** tiene entre **15 y 48 px RMS de ruido** según el
  eje, sobre un nadador que ocupa unos 500 px de largo. Ningún filtro arregla
  eso: no es ruido de alta frecuencia sobre una trayectoria buena, es una
  trayectoria mal estimada.
- Los candidatos a intercambio izquierda/derecha se concentran en los brazos
  (7,5 % de las transiciones en codos, 9,4 % en muñecas) y son prácticamente
  nulos en caderas y rodillas.

Por eso **la rebanada vertical se hace sobre el lado cercano a la cámara (el
izquierdo en este video)**, y el lado lejano se reporta como limitación medida
en lugar de mostrarse como si fuera una medición.

En cambio, hombros y caderas tienen `visibility` por encima de 0,99 en todos los
fotogramas con detección, y su trayectoria es estable: el tronco sirve como
referencia.

Otros dos datos de la misma corrida: la frecuencia de brazada aparece como un
pico claro en **0,47 Hz**, y el tramo continuo con detección más largo dura
**8,0 s** (241 fotogramas), con 15,0 s utilizables sumando los tres tramos más
largos.

### Ángulos del lado cercano

De la misma corrida, en `angulos/informe.md`. Hay ángulo en **608 de los 1055
fotogramas** (57,6 %); el resto no tiene los tres landmarks.

| articulación | marcados | rango sin marcar (mín · mediana · máx) |
|---|---|---|
| codo izq | 27,8 % | 5,2° · 159,9° · 179,9° |
| hombro izq | 16,3 % | 0,0° · 125,8° · 180,0° |
| rodilla izq | 36,0 % | 139,0° · 171,7° · 179,2° |

El motivo de marcado que más pesa es distinto en cada una: `visibility` baja en
la rodilla (33,6 %) y en el codo (16,3 %), intercambio sospechado en el codo
(9,0 %) y en el hombro (7,2 %).

**La rodilla es la medición menos confiable de las tres**: es la que más
marcados acumula (36,0 %, casi todo `visibility` baja) y la que menos rango
recorre (139° a 180°). Ese rango es compatible con la flexión moderada del
batido de crol, así que no se puede decidir desde acá si describe el movimiento
o si el tobillo está mal estimado: hace falta validación contra anotación
manual.

#### La `visibility` no predice si el ángulo es válido

Es el hallazgo más importante de esta etapa, y va en contra de lo que se
esperaría de un puntaje de confianza:

- **El codo toma valores que el cuerpo no puede hacer.** Nueve ángulos (1,5 %)
  caen por debajo de los 35° que deja la flexión máxima del codo —hay valores de
  5,2°, 6,9° y 13,7°— y **siete de esos nueve no están marcados por ningún
  motivo**: la muñeca tenía `visibility` entre **0,40 y 0,57**, cómodamente por
  encima del umbral de reporte de 0,3.
- **La correlación entre la `visibility` mínima de los tres landmarks y el
  ángulo resultante es 0,055**, sobre 603 mediciones. Es decir: ninguna. Saber
  que MediaPipe está seguro de dónde puso el punto no dice nada sobre si el
  ángulo derivado de ese punto tiene sentido.

De ahí salen dos consecuencias para el proyecto:

1. **Las banderas heredadas de los landmarks son necesarias pero no
   suficientes.** Hacen falta criterios que miren la magnitud derivada y no solo
   el dato de origen: si el valor cae fuera del rango que la articulación puede
   recorrer, y si cambia más rápido de lo que el cuerpo puede moverse.
2. **Es un argumento directo a favor de validar contra anotación manual.** Si el
   puntaje de confianza del modelo no separa las mediciones buenas de las
   imposibles, el único juez disponible es un humano marcando fotogramas. Sin
   eso no hay forma de decir cuánto error tiene una medición, que es el aporte
   que este proyecto se propone.

La velocidad angular separa mejor esos casos: el codo se mueve una mediana de
6,1° por fotograma (183 °/s), y seis de los nueve valores imposibles entran o
salen con un salto de entre 47° y 132° en un solo fotograma. Los otros tres
están en el medio de una excursión que dura varios fotogramas, donde no hay
salto que detectar: los dos criterios se complementan, ninguno alcanza solo. La
distribución completa y el costo de cada umbral están en `angulos/informe.md`;
el umbral todavía no está elegido.

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
