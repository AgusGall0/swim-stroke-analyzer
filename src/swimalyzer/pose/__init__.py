"""Estimación de pose con MediaPipe. Sin implementar todavía.

Qué va acá:

- Wrapper de Pose Landmarker en ``RunningMode.VIDEO``, construido a partir de
  las secciones ``modelo`` y ``deteccion`` de la configuración.
- Extracción de los 33 landmarks por fotograma hacia la estructura que
  persiste ``swimalyzer.io``. Este módulo no dibuja ni calcula métricas.
- Esquema fijo de índices de landmarks, en ``swimalyzer.pose.landmarks``.
"""
