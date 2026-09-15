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
video → landmarks a Parquet → filtrado → ángulos bilaterales
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

## Decisiones abiertas

Estas las define Agustín con evidencia, no por defecto. **No las fijes por tu
cuenta**: cuando el código las necesite, proponé opciones con sus trade-offs y
frená.

- **Umbral de `visibility`.** Depende de qué tan ruidoso resulte este video. Se
  elige mirando la distribución real de visibility por landmark, no a ojo.
- **Filtro y frecuencia de corte.** Butterworth de bajo orden es el estándar en
  biomecánica, con corte típico en el entorno de 6 Hz, pero el valor se
  justifica con análisis de residuos sobre los datos propios.
- **Criterio de segmentación de ciclos.** Detección de picos sobre qué señal, con
  qué distancia mínima entre picos, y cómo se valida que los ciclos detectados
  sean reales.
- **Manejo de intercambios izquierda/derecha.** Qué heurística se usa para
  detectarlos.

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
  io/        lectura de video, escritura de Parquet, metadata
  pose/      wrapper de MediaPipe, extracción de landmarks
  signal/    interpolación, filtrado, suavizado
  metrics/   ángulos, segmentación de ciclos, métricas derivadas
  viz/       generación de figuras
tests/
config.yaml
scripts/     descarga del modelo, utilidades
experiments/ los tres scripts originales, conservados como registro
```

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

**Rebanada vertical.** El camino completo descrito arriba en "Alcance", hasta la
figura de curva media normalizada.

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
