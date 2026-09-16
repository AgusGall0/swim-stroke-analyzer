"""Entrada y salida de datos.

- :mod:`swimalyzer.io.video`: lectura de video con OpenCV cuadro a cuadro,
  aplicando el recorte configurado (``video.recorte``) y resolviendo fps y
  resolución reales.
- :mod:`swimalyzer.io.persistencia`: Parquet de landmarks, una fila por
  landmark por fotograma: ``frame, timestamp_ms, landmark_id, x, y, z,
  visibility, presence``. Los fotogramas sin detección quedan registrados con
  ``NaN``, no se saltean.
- :mod:`swimalyzer.io.metadata`: metadata de cada corrida: fps, resolución,
  SHA-256 del modelo y del video, versión del código y parámetros usados.
"""
