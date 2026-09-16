"""Etapa de extracción: video → Parquet de landmarks + metadata de corrida.

Recorre el video una sola vez, corre el detector sobre cada fotograma y
persiste el resultado. No dibuja ni calcula nada derivado: todo lo que venga
después se hace a partir del Parquet.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swimalyzer.config import Configuracion
from swimalyzer.io.metadata import construir_metadata, escribir_metadata
from swimalyzer.io.persistencia import AcumuladorDeLandmarks, escribir_landmarks
from swimalyzer.io.video import InfoVideo, LectorDeVideo
from swimalyzer.pose.deteccion import Detector, DetectorMediaPipe

NOMBRE_LANDMARKS = "landmarks.parquet"
NOMBRE_METADATA = "metadata.json"


@dataclass(frozen=True)
class ResultadoExtraccion:
    """Qué produjo una corrida de extracción."""

    ruta_landmarks: Path
    ruta_metadata: Path
    info: InfoVideo
    fotogramas_procesados: int
    fotogramas_sin_deteccion: tuple[int, ...]
    filas: int
    duracion_s: float

    @property
    def fotogramas_con_deteccion(self) -> int:
        return self.fotogramas_procesados - len(self.fotogramas_sin_deteccion)

    @property
    def tasa_sin_deteccion(self) -> float:
        if self.fotogramas_procesados == 0:
            return 0.0
        return len(self.fotogramas_sin_deteccion) / self.fotogramas_procesados


def _detector_por_defecto(configuracion: Configuracion) -> Detector:
    return DetectorMediaPipe(configuracion.modelo.ruta, configuracion.deteccion)


def extraer(
    video: str | Path,
    destino: str | Path,
    configuracion: Configuracion,
    *,
    fps_forzado: float | None = None,
    crear_detector: Callable[[Configuracion], Detector] | None = None,
    progreso: Callable[[int, int | None], None] | None = None,
    hashear_video: bool = True,
) -> ResultadoExtraccion:
    """Extrae los landmarks de ``video`` y los escribe en ``destino``.

    ``crear_detector`` existe para poder correr el pipeline completo en los
    tests con un detector falso, sin el modelo ``.task``.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    crear_detector = crear_detector or _detector_por_defecto

    comenzado_en = time.perf_counter()
    acumulador = AcumuladorDeLandmarks()
    procesados = 0

    with ExitStack() as recursos:
        lector = recursos.enter_context(
            LectorDeVideo(video, recorte=configuracion.video.recorte, fps_forzado=fps_forzado)
        )
        detector = crear_detector(configuracion)
        cerrar = getattr(detector, "cerrar", None)
        if callable(cerrar):
            recursos.callback(cerrar)

        for indice, fotograma in lector:
            timestamp_ms = lector.timestamp_ms(indice)
            acumulador.agregar(indice, timestamp_ms, detector.detectar(fotograma, timestamp_ms))
            procesados = indice + 1
            if progreso is not None:
                progreso(procesados, lector.info.fotogramas_declarados)

        info = lector.info

    duracion_s = time.perf_counter() - comenzado_en
    ruta_landmarks = escribir_landmarks(acumulador.tabla(), destino / NOMBRE_LANDMARKS)
    sin_deteccion = tuple(acumulador.fotogramas_sin_deteccion)

    resultado_metadata: dict[str, Any] = {
        "archivo_landmarks": ruta_landmarks.name,
        "fotogramas_procesados": procesados,
        "fotogramas_con_deteccion": procesados - len(sin_deteccion),
        "fotogramas_sin_deteccion": len(sin_deteccion),
        "tasa_sin_deteccion": (len(sin_deteccion) / procesados) if procesados else 0.0,
        # Los índices, no solo el total: hay que poder ubicar el hueco en la
        # serie temporal sin volver a leer el Parquet.
        "indices_sin_deteccion": list(sin_deteccion),
        "filas_parquet": len(acumulador),
        "duracion_extraccion_s": round(duracion_s, 3),
    }
    ruta_metadata = escribir_metadata(
        construir_metadata(
            info=info,
            configuracion=configuracion,
            resultado=resultado_metadata,
            hashear_video=hashear_video,
        ),
        destino / NOMBRE_METADATA,
    )

    return ResultadoExtraccion(
        ruta_landmarks=ruta_landmarks,
        ruta_metadata=ruta_metadata,
        info=info,
        fotogramas_procesados=procesados,
        fotogramas_sin_deteccion=sin_deteccion,
        filas=len(acumulador),
        duracion_s=duracion_s,
    )
