# experiments/

Scripts originales del proyecto, escritos entre el 20 y el 23 de marzo de 2026
antes de armar el paquete `swimalyzer`. **Son registro del proceso, no código en
uso:** nada del paquete los importa, no se mantienen, no pasan por lint ni por
tests, y se conservan tal como quedaron (errores incluidos).

Para correr el pipeline actual, ver el README de la raíz.

## Qué era cada uno, en el orden en que se crearon

### 1. `vision_base.py` — prueba de OpenCV con la webcam

Primer paso: comprobar que OpenCV abría la cámara (`VideoCapture(0)`) y mostraba
la imagen en una ventana. En un momento se le agregó MediaPipe y después se le
quitó, cuando ese intento pasó a `apertura_v.py`.

En esa limpieza (commit `0b45b8e`) se borró la rama `else` y la condición quedó
invertida (`if not cap.isOpened():`): en su estado final no muestra nada si la
cámara abre bien.

### 2. `apertura_v.py` — video, pose y ángulo de codo

Arrancó como apertura de un archivo de video con OpenCV y fue creciendo hasta
ser el único prototipo real. Su evolución, según el historial:

1. Lectura de `ejemplo.mp4` cuadro a cuadro con OpenCV.
2. Integración con MediaPipe Pose Landmarker (modelo heavy, `RunningMode.VIDEO`),
   imprimiendo por cuadro si se detectó una persona.
3. Chequeo de la visibility del tobillo izquierdo (landmark 27) con umbral 0.5,
   para ver cómo responde la confianza del modelo cuando el punto no se ve.
4. Estado final: dibuja el esqueleto sobre el video y calcula el ángulo del codo
   derecho (landmarks 12, 14, 16) en píxeles cuando los tres puntos superan
   visibility 0.5, y lo escribe sobre el cuadro.

Limitaciones que el paquete no hereda: no persiste ningún dato (el ángulo se
calcula solo para dibujarlo), saltea en silencio los cuadros sin detección,
redondea a píxel entero antes de calcular el ángulo, mide un solo brazo, y
tiene rutas y umbrales incrustados en el código.

### 3. `pruebamedia.py` — verificación de la instalación de MediaPipe

Tres líneas de `import` de MediaPipe Tasks. Servía para confirmar que el paquete
estaba bien instalado (en particular, que no fallaba por la versión de Python).
No ejecuta nada más.

## Requisitos para correrlos

Esperan `pose_landmarker_heavy.task` y `ejemplo.mp4` en el directorio de
trabajo, y abren ventanas con `cv.imshow`. Ninguno de los dos archivos se
versiona.
