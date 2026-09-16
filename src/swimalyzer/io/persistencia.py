"""Escritura del Parquet de landmarks.

Una fila por landmark por fotograma: ``frame, timestamp_ms, landmark_id, x, y,
z, visibility, presence``. El formato es largo (no una columna por landmark)
porque es el que aguanta sin cambios que el esquema del modelo o el conjunto de
landmarks usados cambien.

**Los fotogramas sin detección no se saltean**: se escriben igual, con sus 33
filas y ``NaN`` en las columnas numéricas. Así la serie temporal queda a paso
constante y el hueco es explícito; la cantidad y los índices de esos fotogramas
también quedan en la metadata de la corrida.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from swimalyzer.pose.deteccion import MuestraLandmark
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS

ESQUEMA_LANDMARKS = pa.schema(
    [
        pa.field("frame", pa.int32(), nullable=False),
        pa.field("timestamp_ms", pa.int64(), nullable=False),
        pa.field("landmark_id", pa.int16(), nullable=False),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("z", pa.float32()),
        pa.field("visibility", pa.float32()),
        pa.field("presence", pa.float32()),
    ]
)

COMPRESION = "snappy"


class AcumuladorDeLandmarks:
    """Junta las filas de la corrida y las vuelca a una tabla de Arrow."""

    def __init__(self) -> None:
        self._frames: list[int] = []
        self._timestamps: list[int] = []
        self._landmark_ids: list[int] = []
        self._x: list[float] = []
        self._y: list[float] = []
        self._z: list[float] = []
        self._visibility: list[float] = []
        self._presence: list[float] = []
        self.fotogramas_sin_deteccion: list[int] = []

    def agregar(
        self, frame: int, timestamp_ms: int, landmarks: list[MuestraLandmark] | None
    ) -> None:
        """Agrega un fotograma. ``landmarks=None`` registra el hueco con NaN."""
        if landmarks is None:
            self.fotogramas_sin_deteccion.append(frame)
            nan = float("nan")
            landmarks = [MuestraLandmark(nan, nan, nan, nan, nan)] * CANTIDAD_LANDMARKS
        for landmark_id, muestra in enumerate(landmarks):
            self._frames.append(frame)
            self._timestamps.append(timestamp_ms)
            self._landmark_ids.append(landmark_id)
            self._x.append(muestra.x)
            self._y.append(muestra.y)
            self._z.append(muestra.z)
            self._visibility.append(muestra.visibility)
            self._presence.append(muestra.presence)

    def __len__(self) -> int:
        return len(self._frames)

    def tabla(self) -> pa.Table:
        return pa.Table.from_arrays(
            [
                pa.array(np.asarray(self._frames, dtype=np.int32)),
                pa.array(np.asarray(self._timestamps, dtype=np.int64)),
                pa.array(np.asarray(self._landmark_ids, dtype=np.int16)),
                pa.array(np.asarray(self._x, dtype=np.float32)),
                pa.array(np.asarray(self._y, dtype=np.float32)),
                pa.array(np.asarray(self._z, dtype=np.float32)),
                pa.array(np.asarray(self._visibility, dtype=np.float32)),
                pa.array(np.asarray(self._presence, dtype=np.float32)),
            ],
            schema=ESQUEMA_LANDMARKS,
        )


def escribir_landmarks(tabla: pa.Table, destino: str | Path) -> Path:
    """Escribe la tabla de landmarks en Parquet y devuelve la ruta."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tabla, destino, compression=COMPRESION)
    return destino


def leer_landmarks(archivo: str | Path) -> pa.Table:
    """Lee un Parquet de landmarks y verifica que tenga el esquema esperado."""
    tabla = pq.read_table(archivo)
    faltantes = [campo.name for campo in ESQUEMA_LANDMARKS if campo.name not in tabla.column_names]
    if faltantes:
        raise ValueError(f"{archivo} no es un Parquet de landmarks: faltan {faltantes}")
    return tabla
