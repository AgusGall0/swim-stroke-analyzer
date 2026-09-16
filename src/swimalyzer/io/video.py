"""Lectura de video fotograma a fotograma con OpenCV.

Entrega los fotogramas ya recortados según ``video.recorte`` y expone las
propiedades reales del archivo (fps y resolución), que van a la metadata de la
corrida. Este módulo no sabe nada de MediaPipe.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2 as cv
import numpy as np

from swimalyzer.config import Recorte

# Por encima de este valor el fps declarado no es creíble y se trata como
# ausente: hay contenedores que informan 1000 o 90000 (la base de tiempo del
# stream) en vez de la tasa real.
FPS_MAXIMO_PLAUSIBLE = 1000.0


class ErrorDeVideo(Exception):
    """El video no se pudo abrir o no se lo puede procesar como está."""


@dataclass(frozen=True)
class InfoVideo:
    """Propiedades del video tal como se lo va a procesar."""

    ruta: Path
    #: fps efectivamente usado para los timestamps.
    fps: float
    #: Lo que informó el contenedor, o ``None`` si no informó nada usable.
    fps_declarado: float | None
    #: Resolución del fotograma que entra a la inferencia (después del recorte).
    ancho: int
    alto: int
    #: Resolución del archivo, antes de recortar.
    ancho_original: int
    alto_original: int
    recorte: Recorte | None
    #: Cantidad de fotogramas que declara el contenedor. Es orientativa: hay
    #: archivos donde no coincide con los que se pueden leer.
    fotogramas_declarados: int | None

    @property
    def fps_fue_forzado(self) -> bool:
        return self.fps_declarado is None


def resolver_fps(fps_contenedor: float, fps_forzado: float | None) -> tuple[float, float | None]:
    """Decide qué fps usar para los timestamps.

    Devuelve ``(fps_a_usar, fps_declarado)``, donde ``fps_declarado`` es
    ``None`` cuando el contenedor no informó un valor usable (0, negativo, NaN
    o absurdamente alto). En ese caso hace falta ``fps_forzado``: sin él no se
    pueden calcular timestamps y se lanza :class:`ErrorDeVideo`, en lugar de
    dividir por cero como hacía el prototipo.
    """
    declarado: float | None = None
    if math.isfinite(fps_contenedor) and 0 < fps_contenedor <= FPS_MAXIMO_PLAUSIBLE:
        declarado = float(fps_contenedor)

    if fps_forzado is not None:
        if not math.isfinite(fps_forzado) or fps_forzado <= 0:
            raise ErrorDeVideo(
                f"el fps indicado tiene que ser un número mayor que 0: {fps_forzado}"
            )
        return float(fps_forzado), declarado

    if declarado is None:
        raise ErrorDeVideo(
            "el contenedor no informa una tasa de fotogramas usable "
            f"(devolvió {fps_contenedor!r}); indicá la real con --fps"
        )
    return declarado, declarado


def validar_recorte(recorte: Recorte, ancho: int, alto: int) -> None:
    """Falla si el recorte se sale del fotograma."""
    if recorte.x + recorte.ancho > ancho or recorte.y + recorte.alto > alto:
        raise ErrorDeVideo(
            f"el recorte ({recorte.x}, {recorte.y}, {recorte.ancho}, {recorte.alto}) "
            f"no entra en un fotograma de {ancho}x{alto}"
        )


def aplicar_recorte(fotograma: np.ndarray, recorte: Recorte | None) -> np.ndarray:
    """Recorta el fotograma. Sin recorte configurado, lo devuelve tal cual."""
    if recorte is None:
        return fotograma
    return fotograma[
        recorte.y : recorte.y + recorte.alto,
        recorte.x : recorte.x + recorte.ancho,
    ]


class LectorDeVideo:
    """Context manager que itera ``(indice, fotograma_bgr)`` ya recortados."""

    def __init__(
        self,
        ruta: str | Path,
        recorte: Recorte | None = None,
        fps_forzado: float | None = None,
    ) -> None:
        self.ruta = Path(ruta)
        self._recorte = recorte
        self._captura = cv.VideoCapture(str(self.ruta))
        if not self._captura.isOpened():
            self._captura.release()
            raise ErrorDeVideo(f"OpenCV no pudo abrir el video: {self.ruta}")

        try:
            fps, fps_declarado = resolver_fps(self._captura.get(cv.CAP_PROP_FPS), fps_forzado)
            ancho_original = int(round(self._captura.get(cv.CAP_PROP_FRAME_WIDTH)))
            alto_original = int(round(self._captura.get(cv.CAP_PROP_FRAME_HEIGHT)))
            if ancho_original <= 0 or alto_original <= 0:
                raise ErrorDeVideo(
                    f"el contenedor no informa una resolución usable: "
                    f"{ancho_original}x{alto_original}"
                )
            if recorte is not None:
                validar_recorte(recorte, ancho_original, alto_original)
            declarados = int(round(self._captura.get(cv.CAP_PROP_FRAME_COUNT)))
        except Exception:
            self._captura.release()
            raise

        self.info = InfoVideo(
            ruta=self.ruta,
            fps=fps,
            fps_declarado=fps_declarado,
            ancho=recorte.ancho if recorte else ancho_original,
            alto=recorte.alto if recorte else alto_original,
            ancho_original=ancho_original,
            alto_original=alto_original,
            recorte=recorte,
            fotogramas_declarados=declarados if declarados > 0 else None,
        )

    def __enter__(self) -> LectorDeVideo:
        return self

    def __exit__(
        self,
        tipo: type[BaseException] | None,
        valor: BaseException | None,
        traza: TracebackType | None,
    ) -> None:
        self.cerrar()

    def cerrar(self) -> None:
        self._captura.release()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        indice = 0
        while True:
            hay_fotograma, fotograma = self._captura.read()
            if not hay_fotograma:
                return
            yield indice, aplicar_recorte(fotograma, self._recorte)
            indice += 1

    def timestamp_ms(self, indice: int) -> int:
        """Timestamp del fotograma ``indice`` en milisegundos enteros.

        MediaPipe en ``RunningMode.VIDEO`` exige timestamps enteros y
        estrictamente crecientes, así que se calculan a partir del índice y el
        fps y no se lee ``CAP_PROP_POS_MSEC`` (que puede repetirse).
        """
        return int(round(indice * 1000.0 / self.info.fps))
