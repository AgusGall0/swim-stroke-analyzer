"""Procesamiento de señal sobre las series temporales de landmarks.

- :mod:`swimalyzer.signal.caracterizacion`: distribución de ``visibility`` por
  landmark, costo de cada umbral, contenido frecuencial, análisis de residuos y
  detección de candidatos a intercambio izquierda/derecha. Produce la evidencia
  con la que se eligen los parámetros; no elige ninguno.
- :mod:`swimalyzer.signal.filtrado`: interpolación de huecos cortos y filtrado
  paso bajo sin desfase, tramo continuo por tramo continuo, más el marcado de
  los intercambios sospechados.

Las trayectorias se manejan **en píxeles**: las coordenadas normalizadas del
Parquet se convierten al cargarlas, con la resolución de inferencia que guarda
la metadata. Medir sobre coordenadas normalizadas deforma distancias y ángulos
por la relación de aspecto.

El filtrado no descarta por ``visibility``: el umbral es criterio de reporte, no
de descarte, y la columna viaja con los datos hasta la etapa de métricas.
"""
