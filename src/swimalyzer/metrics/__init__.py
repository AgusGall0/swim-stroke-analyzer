"""Métricas biomecánicas. Sin implementar todavía.

Qué va acá:

- Ángulos articulares bilaterales, calculados siempre sobre coordenadas en
  píxeles (nunca sobre coordenadas normalizadas, que distorsionan el ángulo
  por la relación de aspecto).
- Segmentación de ciclos de brazada (sección ``segmentacion`` de la
  configuración; el criterio es una decisión abierta).
- Normalización de cada ciclo al 100% y curva media ± desvío estándar.
- Métricas derivadas: frecuencia de brazada, índice de simetría.
"""
