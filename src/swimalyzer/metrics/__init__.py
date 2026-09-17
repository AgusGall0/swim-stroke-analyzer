"""Métricas biomecánicas.

- :mod:`swimalyzer.metrics.angulos`: ángulos articulares del lado cercano a la
  cámara (codo, hombro y rodilla izquierdos), calculados siempre sobre
  coordenadas en píxeles. Cada ángulo hereda las banderas de sus tres
  landmarks: es tan confiable como su peor componente.

Lo que todavía no está:

- Segmentación de ciclos de brazada (sección ``segmentacion`` de la
  configuración; el criterio es una decisión abierta).
- Normalización de cada ciclo al 100% y curva media ± desvío estándar.
- Métricas derivadas: frecuencia de brazada, índice de simetría.
"""
