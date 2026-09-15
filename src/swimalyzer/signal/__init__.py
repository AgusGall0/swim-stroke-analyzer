"""Procesamiento de señal sobre las series temporales de landmarks.

Sin implementar todavía.

Qué va acá:

- Descarte de muestras con visibility por debajo de ``calidad.umbral_visibility``,
  registrando la tasa de descarte por landmark (es un resultado, no un detalle).
- Interpolación de huecos cortos.
- Filtrado paso bajo (sección ``filtrado`` de la configuración). El tipo, el
  orden y la frecuencia de corte son decisiones abiertas: se justifican con
  análisis de residuos sobre los datos propios.
- Detección y corrección de intercambios izquierda/derecha (sección
  ``lateralidad``), también decisión abierta.
"""
