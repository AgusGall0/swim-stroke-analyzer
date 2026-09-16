"""Estimación de pose con MediaPipe.

- :mod:`swimalyzer.pose.landmarks`: esquema fijo de índices de landmarks.
- :mod:`swimalyzer.pose.deteccion`: wrapper de Pose Landmarker en
  ``RunningMode.VIDEO``, construido desde las secciones ``modelo`` y
  ``deteccion`` de la configuración.
- :mod:`swimalyzer.pose.extraccion`: la etapa completa, de video a Parquet más
  metadata.

Este subpaquete no dibuja ni calcula métricas.
"""
