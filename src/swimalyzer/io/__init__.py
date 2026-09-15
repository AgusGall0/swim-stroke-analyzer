"""Entrada y salida de datos. Sin implementar todavía.

Qué va acá:

- Lectura de video con OpenCV cuadro a cuadro, aplicando el recorte
  configurado (``video.recorte``) y obteniendo fps y resolución reales.
- Escritura del Parquet de landmarks, una fila por landmark por fotograma:
  ``frame, timestamp_ms, landmark_id, x, y, z, visibility, presence``.
  Los fotogramas sin detección quedan registrados, no se saltean.
- Metadata de cada corrida: fps, resolución, SHA-256 del modelo, versión del
  código y parámetros de configuración usados.
"""
