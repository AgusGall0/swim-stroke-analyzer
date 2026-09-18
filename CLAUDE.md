# CLAUDE.md

Contexto permanente para trabajar en este repositorio.

## Qué es este proyecto

Sistema de análisis biomecánico de natación por visión computacional. Toma video
de un nadador y produce cinemática cuantitativa de la brazada: ángulos
articulares por fotograma, ciclos de brazada segmentados, y métricas derivadas
como frecuencia de brazada e índice de simetría.

El proyecto es de Agustín Gallo (Ingeniería en Informática, Facultad de
Tecnología y Cs. Aplicadas, U.N.Ca.). Se desarrolla como piloto con intención de
presentarlo en las jornadas de la facultad y, más adelante, como base de un
trabajo de tesis.

**El aporte no es correr un modelo de pose sobre video.** Eso son treinta líneas
usando la API de MediaPipe. El aporte es pasar de detección a medición, y de
medición a medición validada: producir series temporales con criterio
biomecánico, y caracterizar cuánto error tienen.

## Alcance de esta etapa

El objetivo inmediato es **una rebanada vertical**: un camino completo y angosto
que va del video a una figura.

```
video → landmarks a Parquet → filtrado → ángulos del lado cercano (izquierdo)
      → segmentación de ciclos → curva media normalizada al 100% del ciclo
```

Esa figura (curva media ± desvío estándar del ángulo de codo a lo largo del
ciclo) es el entregable de esta etapa. Es la representación canónica en
biomecánica y es lo que se muestra para explicar el proyecto.

**Fuera de alcance por ahora:** otros estilos que no sean crol, análisis en
tiempo real, interfaz gráfica o app móvil, detección automática de estilo,
comparación contra nadador de referencia, calibración métrica píxel a metro,
múltiples nadadores simultáneos. Son ideas válidas, no entran en esta etapa.

Si detectás algo importante fuera de alcance, mencionalo en una línea al final y
seguí. No lo implementes.

## Material de trabajo

El video de desarrollo es un clip de pileta de contracorriente: cámara fija,
plano lateral, nadador estático respecto del encuadre, vista principalmente
subacuática a través de una ventana. 576×512 después de recortar las barras
negras, 30 fps, 35 segundos.

Dos cosas a tener presentes:

- **Resolución baja.** El nadador ocupa unos 500×120 píxeles. Los landmarks
  distales (muñecas, tobillos) van a ser ruidosos. Eso no es un bug.
- **Es material de construcción, no de medición.** Sirve para verificar que el
  pipeline corre de punta a punta. Cualquier número que salga de él es
  provisorio hasta que haya validación contra anotación manual.

El video es de terceros (TikTok). **No se versiona, ni el archivo ni fotogramas
extraídos.** Vive fuera del repo. El material definitivo será grabado por
Agustín con consentimiento de los nadadores.

## Stack

- Python 3.11 o superior. El piso lo fijan las dependencias pinneadas (numpy
  2.4, scipy 1.17 y pandas 3.0 piden 3.11). Aparte, MediaPipe no corre en 3.8:
  usa anotaciones de tipo que rompen con
  `TypeError: 'type' object is not subscriptable`.
- `mediapipe` con Pose Landmarker (modelo `.task`), en `RunningMode.VIDEO`
- `opencv-contrib-python`, **no** `opencv-python`: `mediapipe` ya depende de
  `opencv-contrib-python`, y los dos paquetes escriben el mismo módulo `cv2`.
  Instalados juntos, uno pisa al otro y desinstalar cualquiera rompe el import.
- `numpy`, `scipy`, `pandas`, `pyarrow`, `matplotlib`
- `pydantic` y `pyyaml` para cargar y validar `config.yaml`
- Versiones pinneadas en `pyproject.toml`. No las actualices sin pedirlo.

El modelo `.task` no se versiona: un script lo descarga y verifica su hash.

## Dominio: lo que hay que saber para tocar este código

**Los 33 landmarks de MediaPipe Pose.** El índice identifica el punto y el orden
es fijo. Los relevantes acá:

| Índice | Punto | Índice | Punto |
|---|---|---|---|
| 11 / 12 | hombro izq / der | 23 / 24 | cadera izq / der |
| 13 / 14 | codo izq / der | 25 / 26 | rodilla izq / der |
| 15 / 16 | muñeca izq / der | 27 / 28 | tobillo izq / der |

Del 0 al 10 son cara, del 17 al 22 detalles de mano. No se usan salvo pedido
explícito.

**Izquierda y derecha son del sujeto, no de la cámara.** En vista lateral con
rotación corporal, MediaPipe puede intercambiar los lados cuando el cuerpo pasa
por posiciones ambiguas. Detectar y corregir esos intercambios es parte del
trabajo, no un detalle.

**Coordenadas normalizadas.** `x` e `y` vienen entre 0 y 1, como fracción del
ancho y alto de la imagen. **Los ángulos se calculan siempre sobre coordenadas
convertidas a píxeles** (multiplicando por ancho y alto reales). Calcular un
ángulo directamente sobre coordenadas normalizadas lo distorsiona por la
relación de aspecto: es un error silencioso y está prohibido en este repo.

La componente `z` es profundidad estimada relativa a las caderas. No se usa: es
la menos confiable del modelo y bajo el agua no tiene sentido.

**`visibility`** estima la probabilidad de que el punto esté efectivamente
visible y no ocluido. Es el criterio de calidad del dato. El umbral es
**configurable, nunca una constante en el código**.

La **tasa de fotogramas descartados por visibility baja es un resultado**, no un
problema a esconder. Hay que registrarla y reportarla por landmark y por
condición.

**Glosario de brazada (crol).** Un ciclo completo por brazo tiene cuatro fases:
entrada y extensión (la mano entra al agua y se extiende adelante), agarre
(*catch*, la mano toma el agua y el codo empieza a flexionar), tirón y empuje
(*pull* y *push*, la fase propulsiva bajo el cuerpo), y recobro (*recovery*, el
brazo vuelve adelante fuera del agua).

## Decisiones de diseño ya tomadas

**El pipeline persiste datos, no dibuja.** La extracción produce un Parquet con
una fila por landmark por fotograma: `frame, timestamp_ms, landmark_id, x, y, z,
visibility, presence`. Cualquier visualización se genera después, a partir de
ese archivo. Nunca se calcula una métrica solo para pintarla en pantalla.

**Cada corrida guarda su metadata:** fps, resolución, hash del modelo, versión
del código, parámetros de configuración usados. Sin eso un resultado no es
reproducible.

**Nada de números mágicos.** Umbrales, frecuencias de corte y parámetros de
detección van en `config.yaml`, no incrustados en el código.

**Los fotogramas sin detección se registran, no se saltean en silencio.** La
serie temporal tiene que dejar constancia del hueco.

**Parámetros fijados con el informe de caracterización** (2026-09-16, sobre el
video de desarrollo recortado a 576×324). Están en `config.yaml` con su
justificación; el informe que los respalda se regenera con
`scripts/caracterizar_senal.py`.

- **`calidad.umbral_visibility_reporte: 0.3`.** Es criterio de reporte, no de
  descarte: en vista lateral un umbral global no elige qué fotogramas son malos,
  elige qué miembros existen, y borrar el lado lejano destruiría el resultado
  principal, que es cuantificar cuán poco confiable es.
- **`filtrado`: Butterworth de orden 2 con filtfilt, corte en 3.4 Hz.** Mediana
  del análisis de residuos de Winter sobre este material (óptimo por landmark
  entre 2.0 y 4.25 Hz); el piso de ruido de la PSD arranca en ~4 Hz, así que los
  6 Hz habituales en biomecánica dejarían pasar ruido.
- **`filtrado.hueco_maximo_interpolable_fotogramas: 3`** (0.1 s). Con la brazada
  a 0.55 Hz, interpolar 0.5 s equivale a inventar más de un cuarto de ciclo; los
  huecos más largos quedan en NaN y cortan el tramo.
- **`lateralidad.metodo_correccion_intercambios: ninguno`.** Los intercambios se
  detectan y se marcan con una bandera en el Parquet filtrado, y se informa la
  tasa por par, pero no se corrigen: tocar el dato antes de ver si el artefacto
  llega a la serie de ángulos sería corregir sin evidencia.

**Los ángulos se reportan como ángulo incluido en el vértice**, de 0° a 180°,
con 180° el segmento extendido, y no con la convención anatómica de flexión
(0° extendido). Es la medición directa, sin restarle nada a nada; la convención
de flexión se deriva después sin volver al video. El ángulo es **proyectado** en
el plano de la imagen: con el cuerpo rotado, un valor bajo puede ser flexión o
puede ser escorzo, y con una sola cámara no se distinguen.

**Las banderas heredadas de los landmarks son necesarias pero no suficientes.**
Un ángulo hereda las banderas de sus tres landmarks y queda marcado si alguno
está sin filtrar, interpolado, sospechado de intercambio o con `visibility` por
debajo del umbral de reporte. Eso no alcanza: sobre el video de desarrollo, la
correlación entre la `visibility` mínima de los tres landmarks y el ángulo de
codo resultante es **0,055**, y de los nueve ángulos anatómicamente imposibles
(por debajo de 35°) **siete no quedan marcados por ningún motivo**, porque la
muñeca tenía `visibility` entre 0,40 y 0,57.

La consecuencia es doble: hacen falta criterios que miren la magnitud derivada
—rango anatómico y velocidad angular— y no solo el dato de origen; y, como el
puntaje de confianza del modelo no separa lo bueno de lo imposible, la
validación contra anotación manual deja de ser un lujo y pasa a ser la única
forma de saber cuánto error tiene una medición.

Los dos criterios que se agregaron por esto, con sus valores en `config.yaml`:

- **`calidad.rango_anatomico_grados`: codo [30, 180], hombro [0, 180], rodilla
  [30, 180]**, en grados de ángulo incluido. Salen del rango de movimiento de
  referencia en goniometría (AAOS; Norkin & White), tomando siempre el extremo
  más permisivo: el criterio marca lo imposible, no lo inusual. El del hombro es
  inerte a propósito, porque como ángulo incluido todo el rango es alcanzable.
  El motivo se llama **`fuera_de_rango_en_el_plano_medido`**: un valor fuera de
  rango puede ser un landmark mal estimado o escorzo extremo —la proyección
  puede achicar el ángulo tanto como agrandarlo— y con una sola cámara no se
  distingue cuál. Lo único que se afirma es que esa medición no representa a la
  articulación.
- **`calidad.velocidad_angular_maxima_grados_por_fotograma: 30`** (900 °/s a 30
  fps), uno solo para las tres. Es unas 5 veces el pico que implica la brazada
  (5,6 °/fotograma para una sinusoide a 0,55 Hz con la excursión de codo
  medida), margen que cubre que el agarre es más rápido que el promedio y que la
  señal no es una sinusoide. Marca lo grosero: la mediana del codo ya está por
  encima de ese pico teórico, así que **no estar marcado no es estar limpio**.

**El análisis bilateral no es viable con el video de desarrollo.** Con umbral
0.3 el codo derecho queda sin medición usable en el 96 % de los fotogramas y la
muñeca del lado lejano tiene 15 a 48 px RMS de ruido sobre un nadador de ~500 px.
La rebanada vertical se hace sobre el lado cercano a la cámara (el izquierdo) y
la limitación se reporta, no se esconde.

**El ciclo se corta en la mano más alta del recobro, no en el ángulo de codo.**
Fijado con el barrido del 2026-09-18; el informe que lo respalda se regenera con
`scripts/informe_ciclos.py`.

- **`segmentacion.senal: muneca_y_rel_hombro`**, que es la **altura** de la
  muñeca izquierda **sobre** el hombro izquierdo: `y_hombro − y_muñeca`, positiva
  con la mano arriba. El orden de la resta importa porque en coordenadas de
  imagen la `y` crece hacia abajo; al revés el máximo sería la mano más profunda
  del tirón, que es otro evento y da otros ciclos. Relativa al hombro porque el
  nadador se desplaza dentro del encuadre.
- Las tres señales posicionales que se midieron (muñeca en x, muñeca en y, codo
  en y, todas relativas al hombro) **empatan dentro del ruido**: 19,1 a 19,8° de
  desvío entre ciclos. El desempate no puede ser el decimal, así que gana la que
  corresponde a un **evento nombrado** de la brazada —la mano en lo más alto del
  recobro, antes de la entrada—, que es el punto de referencia habitual en la
  literatura: la fase 0 queda definida en términos biomecánicos y no como el
  máximo de una señal auxiliar.
- **El ángulo de codo está descartado con evidencia**: da ciclos de 1,40 a
  3,07 s porque la serie vive saturada entre 165° y 180° y el detector engancha
  la meseta.
- **`segmentacion.distancia_minima_entre_picos_s: 1.4`.** Entre 1,2 y 1,6 s el
  resultado es idéntico —mismos picos, mismas posiciones—, así que es una meseta
  y 1,4 es el centro. Por debajo de 1,0 s aparece el máximo secundario de cada
  ciclo y la cuenta se duplica.
- **Sin parámetro de prominencia**: en el barrido de 0 a 20 px no cambió ningún
  pico, y un parámetro inerte igual hay que justificarlo después.
- **Un ciclo no cruza un hueco.** Los cortes se buscan tramo continuo por tramo
  continuo: un intervalo que abarque un hueco no es un ciclo, es dos trozos con
  un tiempo indeterminado en el medio.

**La frecuencia de brazada del material es 0,55 Hz**, no los 0,47 Hz que decía
antes. Los 0,47 venían del pico de la PSD del informe de caracterización, que
con ventanas de 8 s tiene una resolución de 0,234 Hz: 0,47 era el bin 2 y no
distinguía 0,47 de 0,55. Los 0,55 salen de medir la duración de los ciclos ya
segmentados (mediana 1,83 s sobre 7 ciclos).

**Los ciclos con mediciones marcadas no se descartan.** Con 7 ciclos, descartar
los que tienen alguna medición marcada deja la muestra en nada y esconde el
problema: la figura saldría limpia porque se le sacó lo sucio, no porque el dato
lo sea. En vez de descartar, la curva media lleva un panel de **cobertura** que
dice, en cada punto del ciclo, qué fracción de las mediciones promediadas está
marcada.

**El recorte no va a `config.yaml`.** `video.recorte` queda en `null`: un
recorte es propiedad del archivo que se procesa, no del método. Para el video de
desarrollo se pasa `--recorte 0 0 576 324` por línea de comandos, y la metadata
de la corrida registra cuál se usó.

## Decisiones abiertas

Estas las define Agustín con evidencia, no por defecto. **No las fijes por tu
cuenta**: cuando el código las necesite, proponé opciones con sus trade-offs y
frená.

En `config.yaml` irían en `null` a propósito: un número provisorio termina en
una figura y nadie recuerda que era provisorio. El código que las use las pide
con `Configuracion.exigir(...)`, que falla mientras sigan en `null`.

Ahora mismo no queda ninguna abierta. Ya resueltas, con su justificación en
"Decisiones de diseño ya tomadas": umbral de `visibility`, filtro y frecuencia
de corte, criterio de hueco corto, manejo de intercambios izquierda/derecha,
rango anatómico de cada articulación, umbral de velocidad angular y criterio de
segmentación de ciclos.

## Convenciones

**Idioma:** código, nombres de variables y funciones en español, igual que el
resto de los proyectos. Docstrings y comentarios en español. Los términos
biomecánicos van en español con el término inglés entre paréntesis la primera
vez que aparecen.

**Mensajes de commit:** primera persona del singular, presente, en español.
Describí qué cambió y por qué, no en qué tarea estabas. Ejemplo: "Agrego
extracción de landmarks a Parquet con registro de fotogramas fallidos".

**Ramas:** una por unidad de trabajo, con prefijo (`feat/`, `fix/`, `chore/`).
Un commit por cambio lógico.

**Estructura:**

```
src/swimalyzer/
  cli.py     subcomandos de línea de comandos
  config.py  carga y validación de config.yaml
  io/        lectura de video, escritura de Parquet, metadata
  pose/      wrapper de MediaPipe, extracción de landmarks
  signal/    interpolación, filtrado, suavizado
  metrics/   ángulos, segmentación de ciclos, métricas derivadas
  viz/       generación de figuras
tests/
config.yaml
scripts/     descarga del modelo, utilidades
experiments/ los tres scripts originales, conservados como registro
modelos/     modelo .task descargado (ignorado por git)
```

Los índices de landmarks son constantes con nombre en
`src/swimalyzer/pose/landmarks.py`, no parámetros de config: son el esquema
fijo del modelo.

Los scripts originales (`vision_base.py`, `pruebamedia.py`, `apertura_v.py`) van
a `experiments/` con un README que explique qué eran. No se borran: son evidencia
del proceso.

## Reglas de trabajo

**No inventes números.** Cualquier cifra que aparezca en documentación o figuras
tiene que venir de una corrida real. Si no podés correrla, decilo.

**Preguntá en vez de asumir.** Si una decisión tiene más de un camino razonable,
frená, explicá las opciones con sus trade-offs, y esperá. Especialmente en
cualquier cosa que toque el cálculo de métricas o el criterio de segmentación.

**Verificá antes de afirmar.** Si vas a decir que algo funciona, corrélo. Si vas
a decir que un landmark es poco confiable, mostrá la distribución de visibility.

**Explicá la biomecánica antes de implementarla.** Cuando escribas una función
que calcula una métrica corporal, decí primero en una o dos líneas qué mide y
por qué se calcula así. Si el razonamiento no cierra, el código tampoco.

**La documentación no puede describir algo que el código no hace.** El README
actual promete análisis en tiempo real y múltiples estilos, y el código hace crol
offline con un brazo. Eso se corrige en la Fase 0.

**No versiones** el video, fotogramas extraídos, el modelo `.task`, ni archivos
de landmarks generados. Todo eso es derivado o de terceros.

## Trabajo pendiente

**Fase 0 — Andamiaje.** Estructura de paquete, `pyproject.toml` con versiones
pinneadas, CLI, `config.yaml`, script de descarga del modelo, tests, CI, LICENSE.
README honesto que describa lo que el proyecto hace hoy.

**Rebanada vertical: terminada.** El camino completo descrito arriba en
"Alcance" llega hasta la figura de curva media normalizada
(`salidas/final/ciclos/curva_media_codo.png`, 7 ciclos, desvío medio 22,3°).

Lo que sigue: **validación contra anotación manual.** Es lo único que separa la
variabilidad del nadador del error del método, y sin eso ningún número que salga
de acá es una medición.

Después de eso se decide el alcance real con evidencia: qué tan bien funciona
MediaPipe en estas condiciones determina si el proyecto apunta a vista lateral
sobre el agua, a comparar condiciones, o a caracterizar las limitaciones del
método.

## Cosas conocidas que no se arreglan por ahora

- **Binario de 29 MB en el historial de git.** El commit `3d86d46` agregó
  `python-3.13.12-amd64.exe` (un instalador de Python para Windows) y `a1b9f44`
  lo borró. Ya no está en el árbol, pero sigue en el historial e infla cada
  clon. Limpiarlo requiere reescribir el historial (`git filter-repo` o
  similar) y forzar el push, lo que obliga a re-clonar. Fuera de alcance hasta
  que se decida explícitamente.

## Contexto que conviene tener presente

Agustín ya trabajó con MediaPipe Tasks y modelos TFLite en un proyecto Android
previo, integrando un modelo de detección de objetos ajeno en la app de ejemplo
de MediaPipe. O sea que la API le resulta familiar. Lo nuevo acá es todo lo que
viene después de la inferencia: persistencia, procesamiento de señal, métricas
biomecánicas y validación.
