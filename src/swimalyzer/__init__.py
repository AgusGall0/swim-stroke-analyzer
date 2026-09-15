"""swimalyzer: análisis biomecánico de la brazada de crol a partir de video.

Camino previsto (rebanada vertical, todavía sin implementar):

    video → landmarks a Parquet → filtrado → ángulos bilaterales
          → segmentación de ciclos → curva media normalizada al 100% del ciclo

Subpaquetes:

- ``io``: lectura de video, escritura de Parquet, metadata de corrida.
- ``pose``: wrapper de MediaPipe Pose Landmarker y extracción de landmarks.
- ``signal``: interpolación de huecos, filtrado y suavizado.
- ``metrics``: ángulos articulares, segmentación de ciclos, métricas derivadas.
- ``viz``: generación de figuras a partir de datos ya persistidos.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("swimalyzer")
except PackageNotFoundError:  # paquete sin instalar (por ejemplo, corriendo desde src/)
    __version__ = "desconocida"
