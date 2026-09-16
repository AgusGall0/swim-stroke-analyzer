"""Wrapper de MediaPipe Pose Landmarker en ``RunningMode.VIDEO``.

Este módulo solo detecta: no dibuja, no filtra y no calcula métricas. Devuelve
las coordenadas **normalizadas** tal como las da el modelo, sin redondear a
píxel entero: redondear acá perdería resolución angular de forma irreversible.
La conversión a píxeles la hace la etapa que calcula ángulos, con el ancho y
alto reales que quedan registrados en la metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

import cv2 as cv
import mediapipe as mp
import numpy as np

from swimalyzer.config import Deteccion
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS

#: Valor que se registra cuando el modelo no informa visibility o presence.
SIN_DATO = float("nan")


@dataclass(frozen=True, slots=True)
class MuestraLandmark:
    """Un landmark en un fotograma, en coordenadas normalizadas (0 a 1)."""

    x: float
    y: float
    z: float
    visibility: float
    presence: float


class Detector(Protocol):
    """Lo que la extracción necesita de un detector de pose.

    Está como Protocol para poder correr el pipeline en los tests con un
    detector falso, sin el modelo ``.task`` ni MediaPipe.
    """

    def detectar(
        self, fotograma_bgr: np.ndarray, timestamp_ms: int
    ) -> list[MuestraLandmark] | None:
        """Devuelve los landmarks del fotograma, o ``None`` si no hubo detección."""
        ...


def _numero(valor: float | None) -> float:
    return SIN_DATO if valor is None else float(valor)


class DetectorMediaPipe:
    """Pose Landmarker sobre video, construido desde la configuración."""

    def __init__(self, ruta_modelo: str | Path, deteccion: Deteccion) -> None:
        ruta_modelo = Path(ruta_modelo)
        if not ruta_modelo.is_file():
            raise FileNotFoundError(
                f"no está el modelo en {ruta_modelo}; descargalo con "
                "python scripts/descargar_modelo.py"
            )
        opciones = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ruta_modelo)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=deteccion.num_poses,
            min_pose_detection_confidence=deteccion.min_pose_detection_confidence,
            min_pose_presence_confidence=deteccion.min_pose_presence_confidence,
            min_tracking_confidence=deteccion.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opciones)

    def __enter__(self) -> DetectorMediaPipe:
        return self

    def __exit__(
        self,
        tipo: type[BaseException] | None,
        valor: BaseException | None,
        traza: TracebackType | None,
    ) -> None:
        self.cerrar()

    def cerrar(self) -> None:
        self._landmarker.close()

    def detectar(
        self, fotograma_bgr: np.ndarray, timestamp_ms: int
    ) -> list[MuestraLandmark] | None:
        imagen = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv.cvtColor(fotograma_bgr, cv.COLOR_BGR2RGB),
        )
        resultado = self._landmarker.detect_for_video(imagen, timestamp_ms)
        if not resultado.pose_landmarks:
            return None
        # num_poses vale 1 en este proyecto: varios nadadores está fuera de alcance.
        landmarks = resultado.pose_landmarks[0]
        if len(landmarks) != CANTIDAD_LANDMARKS:
            raise RuntimeError(
                f"el modelo devolvió {len(landmarks)} landmarks y se esperaban {CANTIDAD_LANDMARKS}"
            )
        return [
            MuestraLandmark(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=_numero(landmark.visibility),
                presence=_numero(landmark.presence),
            )
            for landmark in landmarks
        ]
