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

#: Los doce landmarks del tronco y las extremidades que usa el análisis.
LANDMARKS_DE_INTERES: tuple[int, ...] = (
    HOMBRO_IZQ,
    HOMBRO_DER,
    CODO_IZQ,
    CODO_DER,
    MUNECA_IZQ,
    MUNECA_DER,
    CADERA_IZQ,
    CADERA_DER,
    RODILLA_IZQ,
    RODILLA_DER,
    TOBILLO_IZQ,
    TOBILLO_DER,
)

#: Pares homólogos (izquierdo, derecho): los candidatos a intercambio
#: izquierda/derecha cuando el cuerpo pasa por posiciones ambiguas.
PARES_HOMOLOGOS: tuple[tuple[int, int], ...] = (
    (HOMBRO_IZQ, HOMBRO_DER),
    (CODO_IZQ, CODO_DER),
    (MUNECA_IZQ, MUNECA_DER),
    (CADERA_IZQ, CADERA_DER),
    (RODILLA_IZQ, RODILLA_DER),
    (TOBILLO_IZQ, TOBILLO_DER),
)

#: Nombre legible de cada landmark de interés, para informes y figuras.
NOMBRES: dict[int, str] = {
    HOMBRO_IZQ: "hombro_izq",
    HOMBRO_DER: "hombro_der",
    CODO_IZQ: "codo_izq",
    CODO_DER: "codo_der",
    MUNECA_IZQ: "muneca_izq",
    MUNECA_DER: "muneca_der",
    CADERA_IZQ: "cadera_izq",
    CADERA_DER: "cadera_der",
    RODILLA_IZQ: "rodilla_izq",
    RODILLA_DER: "rodilla_der",
    TOBILLO_IZQ: "tobillo_izq",
    TOBILLO_DER: "tobillo_der",
}


def nombre(landmark_id: int) -> str:
    """Nombre legible del landmark, o ``landmark_NN`` si no es uno de interés."""
    return NOMBRES.get(landmark_id, f"landmark_{landmark_id:02d}")
