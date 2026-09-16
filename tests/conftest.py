"""Rutas y utilidades compartidas por los tests."""

import json
from pathlib import Path

import numpy as np

from swimalyzer.io.persistencia import AcumuladorDeLandmarks, escribir_landmarks
from swimalyzer.pose.deteccion import MuestraLandmark
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS

RAIZ_DEL_REPO = Path(__file__).resolve().parent.parent
CONFIG_DEL_REPO = RAIZ_DEL_REPO / "config.yaml"

FPS = 30.0
ANCHO, ALTO = 576, 324


def escribir_corrida(directorio, x, y, visibility, detectado=None):
    """Arma una corrida de extracción en disco desde matrices (fotogramas, landmarks).

    ``x`` e ``y`` van normalizadas, como las guarda la extracción.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    fotogramas = x.shape[0]
    detectado = np.ones(fotogramas, bool) if detectado is None else detectado
    acumulador = AcumuladorDeLandmarks()
    for frame in range(fotogramas):
        if not detectado[frame]:
            acumulador.agregar(frame, int(frame * 1000 / FPS), None)
            continue
        acumulador.agregar(
            frame,
            int(frame * 1000 / FPS),
            [
                MuestraLandmark(
                    x=float(x[frame, landmark]),
                    y=float(y[frame, landmark]),
                    z=0.0,
                    visibility=float(visibility[frame, landmark]),
                    presence=1.0,
                )
                for landmark in range(CANTIDAD_LANDMARKS)
            ],
        )
    escribir_landmarks(acumulador.tabla(), directorio / "landmarks.parquet")
    (directorio / "metadata.json").write_text(
        json.dumps(
            {
                "etapa": "extraccion",
                "video": {
                    "archivo": "prueba.mp4",
                    "fps": FPS,
                    "resolucion_inferencia": [ANCHO, ALTO],
                },
            }
        ),
        encoding="utf-8",
    )
    return directorio
