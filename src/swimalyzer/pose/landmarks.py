"""Índices de los landmarks de MediaPipe Pose usados en el análisis.

El índice identifica el punto y el orden lo fija el modelo: no son parámetros
ajustables, por eso viven en código y no en ``config.yaml``. Si cambian, cambió
el modelo.

Izquierda y derecha son del sujeto, no de la cámara.

Del 0 al 10 son cara y del 17 al 22 detalles de mano; no se usan salvo pedido
explícito.
"""

HOMBRO_IZQ = 11
HOMBRO_DER = 12
CODO_IZQ = 13
CODO_DER = 14
MUNECA_IZQ = 15
MUNECA_DER = 16
CADERA_IZQ = 23
CADERA_DER = 24
RODILLA_IZQ = 25
RODILLA_DER = 26
TOBILLO_IZQ = 27
TOBILLO_DER = 28

CANTIDAD_LANDMARKS = 33
