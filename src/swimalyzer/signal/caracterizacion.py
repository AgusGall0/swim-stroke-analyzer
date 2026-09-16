"""Caracterización de la señal cruda, antes de decidir cómo filtrarla.

Este módulo no elige umbrales ni frecuencias de corte: produce la evidencia con
la que se eligen. Todo sale del Parquet de landmarks y su metadata, sin volver
a procesar el video.

Las trayectorias se devuelven **en píxeles**. Las coordenadas del Parquet están
normalizadas (0 a 1) respecto del ancho y del alto, que en este video son
distintos: medir sobre ellas distorsiona cualquier ángulo o distancia por la
relación de aspecto. La conversión se hace acá, una sola vez, con la resolución
de inferencia que quedó registrada en la metadata de la corrida.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sg

from swimalyzer.io.persistencia import leer_landmarks
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS


@dataclass(frozen=True)
class SeriesDeLandmarks:
    """Trayectorias de una corrida, en píxeles, a paso constante de fotograma.

    Las filas son fotogramas y las columnas landmarks. Un fotograma sin
    detección es una fila entera de ``NaN``: el hueco está, no se rellena.
    """

    x: np.ndarray
    y: np.ndarray
    visibility: np.ndarray
    #: Marca de tiempo de cada fotograma, tal como la guardó la extracción.
    timestamp_ms: np.ndarray
    fps: float
    ancho: int
    alto: int
    origen: Path

    @property
    def cantidad_fotogramas(self) -> int:
        return self.x.shape[0]

    @property
    def detectado(self) -> np.ndarray:
        """Máscara de fotogramas en los que hubo detección."""
        return ~np.isnan(self.x).all(axis=1)

    @property
    def duracion_s(self) -> float:
        return self.cantidad_fotogramas / self.fps

    def usable(self, umbral_visibility: float) -> np.ndarray:
        """Máscara ``(fotogramas, landmarks)`` de muestras que pasan el umbral."""
        with np.errstate(invalid="ignore"):
            return ~np.isnan(self.visibility) & (self.visibility >= umbral_visibility)


def cargar_series(directorio: str | Path) -> SeriesDeLandmarks:
    """Lee ``landmarks.parquet`` y ``metadata.json`` de una corrida."""
    directorio = Path(directorio)
    metadata = json.loads((directorio / "metadata.json").read_text(encoding="utf-8"))
    ancho, alto = metadata["video"]["resolucion_inferencia"]
    fps = float(metadata["video"]["fps"])

    tabla = leer_landmarks(directorio / "landmarks.parquet")
    datos = {nombre: np.asarray(tabla.column(nombre)) for nombre in tabla.column_names}
    fotogramas = int(datos["frame"].max()) + 1
    forma = (fotogramas, CANTIDAD_LANDMARKS)
    indice = (datos["frame"].astype(np.int64), datos["landmark_id"].astype(np.int64))

    def matriz(columna: str) -> np.ndarray:
        salida = np.full(forma, np.nan, dtype=np.float64)
        salida[indice] = datos[columna].astype(np.float64)
        return salida

    timestamps = np.zeros(fotogramas, dtype=np.int64)
    timestamps[datos["frame"].astype(np.int64)] = datos["timestamp_ms"].astype(np.int64)

    return SeriesDeLandmarks(
        x=matriz("x") * ancho,
        y=matriz("y") * alto,
        visibility=matriz("visibility"),
        timestamp_ms=timestamps,
        fps=fps,
        ancho=ancho,
        alto=alto,
        origen=directorio,
    )


# --- distribución de visibility --------------------------------------------

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def percentiles_de_visibility(
    series: SeriesDeLandmarks, landmark_id: int, percentiles: tuple[int, ...] = PERCENTILES
) -> dict[str, float]:
    """Distribución de visibility de un landmark, sobre fotogramas detectados.

    Los fotogramas sin detección no entran: no tienen un valor bajo, no tienen
    valor. Se los cuenta aparte, en la tasa de fotogramas sin detección.
    """
    valores = series.visibility[series.detectado, landmark_id]
    valores = valores[~np.isnan(valores)]
    if valores.size == 0:
        return {"n": 0}
    resumen = {
        f"p{p}": float(valor)
        for p, valor in zip(percentiles, np.percentile(valores, percentiles), strict=True)
    }
    resumen.update(
        {
            "n": int(valores.size),
            "media": float(valores.mean()),
            "desvio": float(valores.std(ddof=1)) if valores.size > 1 else 0.0,
            "minimo": float(valores.min()),
            "maximo": float(valores.max()),
        }
    )
    return resumen


def descarte_por_umbral(
    series: SeriesDeLandmarks, landmark_id: int, umbral: float
) -> dict[str, float]:
    """Qué fracción de fotogramas queda sin dato usable con ese umbral.

    Se informan dos denominadores porque responden preguntas distintas:
    ``sobre_detectados`` mide cuánto descarta el umbral en sí, y ``sobre_todos``
    mide cuánta serie temporal queda, contando también los fotogramas donde no
    hubo ninguna detección.
    """
    detectado = series.detectado
    visibility = series.visibility[:, landmark_id]
    with np.errstate(invalid="ignore"):
        pasa = detectado & ~np.isnan(visibility) & (visibility >= umbral)
    detectados = int(detectado.sum())
    return {
        "umbral": float(umbral),
        "descartados_sobre_todos": float(1 - pasa.sum() / series.cantidad_fotogramas),
        "descartados_sobre_detectados": (float(1 - pasa.sum() / detectados) if detectados else 1.0),
        "fotogramas_usables": int(pasa.sum()),
    }


# --- tramos continuos ------------------------------------------------------


def tramos_continuos(mascara: np.ndarray, largo_minimo: int = 1) -> list[tuple[int, int]]:
    """Tramos ``[inicio, fin)`` de ``True`` seguidos, de al menos ``largo_minimo``."""
    tramos: list[tuple[int, int]] = []
    inicio: int | None = None
    for indice, valor in enumerate(mascara):
        if valor and inicio is None:
            inicio = indice
        elif not valor and inicio is not None:
            if indice - inicio >= largo_minimo:
                tramos.append((inicio, indice))
            inicio = None
    if inicio is not None and len(mascara) - inicio >= largo_minimo:
        tramos.append((inicio, len(mascara)))
    return tramos


def huecos(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Tramos ``[inicio, fin)`` sin dato. Son los que habría que interpolar."""
    return tramos_continuos(~mascara)


# --- contenido frecuencial -------------------------------------------------


def espectro(
    serie: np.ndarray, fps: float, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Densidad espectral de potencia (Welch) de una trayectoria sin huecos.

    Se le resta la media: el interés está en cómo se mueve el punto, no en
    dónde está dentro del encuadre.
    """
    if np.isnan(serie).any():
        raise ValueError("el espectro se calcula sobre un tramo sin huecos")
    nperseg = min(nperseg or 128, serie.size)
    return sg.welch(serie - serie.mean(), fs=fps, nperseg=nperseg, noverlap=nperseg // 2)


def frecuencia_de_potencia_acumulada(
    frecuencias: np.ndarray, potencia: np.ndarray, fraccion: float
) -> float:
    """Frecuencia por debajo de la cual está ``fraccion`` de la potencia."""
    acumulada = np.cumsum(potencia) / potencia.sum()
    return float(frecuencias[int(np.searchsorted(acumulada, fraccion))])


CORTES_POR_DEFECTO = np.arange(0.5, 14.01, 0.25)


@dataclass(frozen=True)
class Residuos:
    """Resultado del análisis de residuos de Winter para una trayectoria."""

    cortes_hz: np.ndarray
    residuo_rms_px: np.ndarray
    ruido_estimado_px: float
    corte_optimo_hz: float
    pendiente: float


def analisis_de_residuos(
    serie: np.ndarray,
    fps: float,
    orden: int = 2,
    cortes_hz: np.ndarray = CORTES_POR_DEFECTO,
    ajuste_desde_hz: float = 8.0,
) -> Residuos:
    """Residuo RMS entre la señal cruda y la filtrada, para varios cortes.

    Es el método clásico de Winter para justificar una frecuencia de corte.
    Cuanto más alto el corte, menos le saca el filtro a la señal y más chico es
    el residuo. Si por encima de cierta frecuencia lo único que se está
    quitando es ruido, esa caída es lineal; la recta ajustada en esa zona,
    extrapolada a corte cero, estima cuánto ruido hay en total. El corte donde
    el residuo real iguala esa estimación es el que saca el ruido sin empezar a
    comerse la señal.

    Que este método devuelva un número **no lo convierte en la decisión**: hay
    que mirar si la curva tiene de verdad una zona lineal.
    """
    if np.isnan(serie).any():
        raise ValueError("el análisis de residuos se hace sobre un tramo sin huecos")
    nyquist = fps / 2
    residuos = []
    for corte in cortes_hz:
        b, a = sg.butter(orden, corte / nyquist, btype="low")
        residuos.append(float(np.sqrt(np.mean((serie - sg.filtfilt(b, a, serie)) ** 2))))
    residuo_rms = np.asarray(residuos)

    zona_lineal = cortes_hz >= ajuste_desde_hz
    pendiente, ordenada = np.polyfit(cortes_hz[zona_lineal], residuo_rms[zona_lineal], 1)
    por_debajo = np.where(residuo_rms <= ordenada)[0]
    corte_optimo = float(cortes_hz[por_debajo[0]]) if por_debajo.size else float("nan")
    return Residuos(
        cortes_hz=np.asarray(cortes_hz),
        residuo_rms_px=residuo_rms,
        ruido_estimado_px=float(ordenada),
        corte_optimo_hz=corte_optimo,
        pendiente=float(pendiente),
    )


# --- intercambios izquierda / derecha --------------------------------------


@dataclass(frozen=True)
class CandidatosAIntercambio:
    """Fotogramas donde asignar al revés los landmarks homólogos explicaría mejor el movimiento."""

    izquierdo: int
    derecho: int
    transiciones_evaluadas: int
    fotogramas: np.ndarray
    margen_px: np.ndarray
    separacion_px: np.ndarray
    cruces_en_x: int


def candidatos_a_intercambio(
    series: SeriesDeLandmarks,
    izquierdo: int,
    derecho: int,
    margen_minimo_px: float = 20.0,
    separacion_minima_px: float = 20.0,
) -> CandidatosAIntercambio:
    """Busca intercambios izquierda/derecha entre dos landmarks homólogos.

    Entre dos fotogramas seguidos se comparan dos hipótesis: que cada punto
    siguió al suyo, o que el modelo los cambió de lado. Se mide el
    desplazamiento total que implica cada una y se marca el fotograma cuando la
    hipótesis de intercambio implica bastante menos movimiento. Un cuerpo no se
    teletransporta: si cambiar las etiquetas explica mejor lo observado, lo más
    probable es que las etiquetas estén cambiadas.

    Solo se marcan pares que además estaban bien separados en los dos
    fotogramas: cuando los dos puntos están casi encimados, cuál es cuál no
    cambia ninguna medición y el criterio se vuelve ruido.

    Esto es un **detector de candidatos**, no una corrección: qué heurística se
    usa para corregir es una decisión abierta.
    """
    izq = np.stack([series.x[:, izquierdo], series.y[:, izquierdo]], axis=1)
    der = np.stack([series.x[:, derecho], series.y[:, derecho]], axis=1)
    detectado = series.detectado
    evaluables = detectado[:-1] & detectado[1:]

    def distancia(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a - b, axis=1)

    costo_identidad = distancia(izq[1:], izq[:-1]) + distancia(der[1:], der[:-1])
    costo_intercambio = distancia(izq[1:], der[:-1]) + distancia(der[1:], izq[:-1])
    margen = costo_identidad - costo_intercambio

    separacion = distancia(izq, der)
    separados = (separacion[:-1] >= separacion_minima_px) & (separacion[1:] >= separacion_minima_px)
    with np.errstate(invalid="ignore"):
        marcados = evaluables & separados & (margen >= margen_minimo_px)
    indices = np.where(marcados)[0] + 1  # el fotograma en el que aparece el cambio

    diferencia_x = (series.x[:, izquierdo] - series.x[:, derecho])[detectado]
    signos = np.sign(diferencia_x)
    cruces = int((signos[1:] * signos[:-1] < 0).sum())

    return CandidatosAIntercambio(
        izquierdo=izquierdo,
        derecho=derecho,
        transiciones_evaluadas=int(evaluables.sum()),
        fotogramas=indices,
        margen_px=margen[marcados],
        separacion_px=separacion[indices],
        cruces_en_x=cruces,
    )
