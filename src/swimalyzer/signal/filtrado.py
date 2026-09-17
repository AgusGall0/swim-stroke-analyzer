"""Interpolación de huecos cortos y filtrado paso bajo de las trayectorias.

El orden de las operaciones importa. Primero se rellenan los huecos cortos, que
si no cortarían el tramo en dos; después se filtra cada tramo continuo por
separado, porque filtrar a través de un hueco mezclaría dos trozos de
movimiento que no son consecutivos.

**No hay gate por visibility.** Se filtra todo lo que MediaPipe detectó y la
visibility viaja como columna hasta la etapa de métricas, que la usa para marcar
mediciones poco confiables e informar cobertura. Descartar por visibility en
esta etapa no elegiría qué fotogramas son malos sino qué miembros existen: en
vista lateral la visibility discrimina lado cercano de lado lejano, y perder el
lado lejano borraría justamente lo que hay que cuantificar.

Los intercambios izquierda/derecha se detectan y se marcan, no se corrigen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal as sg

from swimalyzer.config import Configuracion, ErrorDeConfiguracion
from swimalyzer.io.metadata import encabezado_de_corrida, escribir_metadata, sha256_de_archivo
from swimalyzer.io.persistencia import COMPRESION
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS, PARES_HOMOLOGOS, nombre
from swimalyzer.signal.caracterizacion import (
    SeriesDeLandmarks,
    candidatos_a_intercambio,
    cargar_series,
    tramos_continuos,
)

NOMBRE_FILTRADO = "landmarks_filtrados.parquet"
NOMBRE_METADATA = "metadata_filtrado.json"

TIPOS_SOPORTADOS = ("butterworth",)

ESQUEMA_FILTRADO = pa.schema(
    [
        pa.field("frame", pa.int32(), nullable=False),
        pa.field("timestamp_ms", pa.int64(), nullable=False),
        pa.field("landmark_id", pa.int16(), nullable=False),
        # En píxeles, no normalizadas: el nombre lo dice para que nadie calcule
        # un ángulo sobre coordenadas normalizadas por distracción.
        pa.field("x_px", pa.float32()),
        pa.field("y_px", pa.float32()),
        pa.field("visibility", pa.float32()),
        pa.field("interpolado", pa.bool_(), nullable=False),
        pa.field("filtrado", pa.bool_(), nullable=False),
        pa.field("intercambio_sospechado", pa.bool_(), nullable=False),
    ]
)


class ErrorDeFiltrado(Exception):
    """Los parámetros de filtrado no se pueden aplicar a estos datos."""


@dataclass(frozen=True)
class SeriesFiltradas(SeriesDeLandmarks):
    """Trayectorias ya filtradas, en píxeles, con las banderas de cada muestra.

    Las banderas son por muestra (fotograma, landmark) y dicen qué le pasó al
    dato: si se rellenó por interpolación, si llegó a pasar por el filtro y si
    el fotograma quedó sospechado de intercambio izquierda/derecha. Viajan con
    los datos hasta las métricas, que las heredan.
    """

    interpolado: np.ndarray
    filtrado: np.ndarray
    intercambio_sospechado: np.ndarray


def cargar_filtrado(directorio: str | Path) -> SeriesFiltradas:
    """Lee ``landmarks_filtrados.parquet`` y su metadata.

    Las coordenadas ya vienen en píxeles del Parquet filtrado: acá no se
    convierte nada. Los fps y la resolución salen de la metadata de la
    extracción, que el filtrado guarda anidada dentro de la suya.
    """
    directorio = Path(directorio)
    archivo = directorio / NOMBRE_FILTRADO
    if not archivo.is_file():
        raise ErrorDeFiltrado(
            f"no hay landmarks filtrados en {directorio} (falta {NOMBRE_FILTRADO})"
        )

    metadata = json.loads((directorio / NOMBRE_METADATA).read_text(encoding="utf-8"))
    extraccion = (metadata.get("origen") or {}).get("metadata_extraccion")
    if not extraccion:
        # Si el filtrado se escribió en otro directorio y la metadata de origen
        # no quedó anidada, la de la extracción tiene que estar al lado.
        extraccion = json.loads((directorio / "metadata.json").read_text(encoding="utf-8"))
    ancho, alto = extraccion["video"]["resolucion_inferencia"]
    fps = float(extraccion["video"]["fps"])

    tabla = pq.read_table(archivo)
    faltantes = [campo.name for campo in ESQUEMA_FILTRADO if campo.name not in tabla.column_names]
    if faltantes:
        raise ErrorDeFiltrado(
            f"{archivo} no es un Parquet de landmarks filtrados: faltan {faltantes}"
        )

    datos = {nombre: np.asarray(tabla.column(nombre)) for nombre in tabla.column_names}
    fotogramas = int(datos["frame"].max()) + 1
    forma = (fotogramas, CANTIDAD_LANDMARKS)
    indice = (datos["frame"].astype(np.int64), datos["landmark_id"].astype(np.int64))

    def matriz(columna: str, relleno: Any, tipo: Any) -> np.ndarray:
        salida = np.full(forma, relleno, dtype=tipo)
        salida[indice] = datos[columna].astype(tipo)
        return salida

    timestamps = np.zeros(fotogramas, dtype=np.int64)
    timestamps[datos["frame"].astype(np.int64)] = datos["timestamp_ms"].astype(np.int64)

    return SeriesFiltradas(
        x=matriz("x_px", np.nan, np.float64),
        y=matriz("y_px", np.nan, np.float64),
        visibility=matriz("visibility", np.nan, np.float64),
        timestamp_ms=timestamps,
        fps=fps,
        ancho=ancho,
        alto=alto,
        origen=directorio,
        interpolado=matriz("interpolado", False, bool),
        filtrado=matriz("filtrado", False, bool),
        intercambio_sospechado=matriz("intercambio_sospechado", False, bool),
    )


def longitud_minima_para_filtfilt(orden: int) -> int:
    """Muestras mínimas que necesita ``filtfilt`` con un Butterworth de ese orden.

    ``filtfilt`` extiende la señal en los bordes para no inventar un transitorio,
    y no puede extenderla más de lo que mide. Un tramo más corto que esto se deja
    sin filtrar y queda marcado como tal: es preferible a rellenar el tramo con
    un filtro distinto y no decirlo.
    """
    return 3 * (orden + 1) + 1


def interpolar_huecos_cortos(serie: np.ndarray, maximo: int) -> tuple[np.ndarray, np.ndarray]:
    """Rellena linealmente los huecos de hasta ``maximo`` muestras.

    Devuelve la serie y la máscara de lo que se rellenó. Los huecos más largos
    quedan en ``NaN`` y cortan el tramo. Un hueco pegado al principio o al final
    de la serie tampoco se rellena: ahí no hay dos valores entre los cuales
    interpolar, y prolongar el último valor conocido sería inventar movimiento.
    """
    salida = serie.astype(np.float64, copy=True)
    interpolado = np.zeros(serie.size, dtype=bool)
    valido = ~np.isnan(serie)
    if maximo <= 0 or valido.sum() < 2:
        return salida, interpolado

    indices = np.arange(serie.size)
    for inicio, fin in tramos_continuos(~valido):
        if fin - inicio > maximo or inicio == 0 or fin == serie.size:
            continue
        salida[inicio:fin] = np.interp(indices[inicio:fin], indices[valido], serie[valido])
        interpolado[inicio:fin] = True
    return salida, interpolado


def filtrar_trayectoria(
    serie: np.ndarray, fps: float, orden: int, frecuencia_corte_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Butterworth paso bajo con ``filtfilt``, tramo continuo por tramo continuo.

    ``filtfilt`` pasa el filtro en los dos sentidos: el resultado no tiene
    desfase, que es lo que hace falta para que un pico de la señal filtrada caiga
    en el mismo fotograma que el del movimiento real.

    Devuelve la serie filtrada y la máscara de las muestras que efectivamente
    pasaron por el filtro.
    """
    nyquist = fps / 2
    if not 0 < frecuencia_corte_hz < nyquist:
        raise ErrorDeFiltrado(
            f"la frecuencia de corte ({frecuencia_corte_hz} Hz) tiene que estar entre 0 y "
            f"el Nyquist del video ({nyquist} Hz)"
        )

    salida = serie.astype(np.float64, copy=True)
    filtrado = np.zeros(serie.size, dtype=bool)
    minimo = longitud_minima_para_filtfilt(orden)
    b, a = sg.butter(orden, frecuencia_corte_hz / nyquist, btype="low")
    for inicio, fin in tramos_continuos(~np.isnan(serie)):
        if fin - inicio < minimo:
            continue
        salida[inicio:fin] = sg.filtfilt(b, a, serie[inicio:fin])
        filtrado[inicio:fin] = True
    return salida, filtrado


def marcar_intercambios(
    series: SeriesDeLandmarks, margen_minimo_px: float, separacion_minima_px: float
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Marca los fotogramas sospechados de intercambio en los pares homólogos.

    La bandera queda en los dos landmarks del par y solo en el fotograma donde
    aparece el cambio, no en todo el intervalo que dure: el detector compara
    fotogramas consecutivos y no sabe cuándo volvió a su lugar.
    """
    banderas = np.zeros((series.cantidad_fotogramas, CANTIDAD_LANDMARKS), dtype=bool)
    tasas: dict[str, dict[str, Any]] = {}
    for izquierdo, derecho in PARES_HOMOLOGOS:
        candidatos = candidatos_a_intercambio(
            series,
            izquierdo,
            derecho,
            margen_minimo_px=margen_minimo_px,
            separacion_minima_px=separacion_minima_px,
        )
        banderas[candidatos.fotogramas, izquierdo] = True
        banderas[candidatos.fotogramas, derecho] = True
        par = f"{nombre(izquierdo)}/{nombre(derecho)}"
        tasas[par] = {
            "transiciones_evaluadas": candidatos.transiciones_evaluadas,
            "fotogramas_marcados": int(candidatos.fotogramas.size),
            "tasa": float(candidatos.fotogramas.size / max(candidatos.transiciones_evaluadas, 1)),
            "margen_mediano_px": (
                float(np.median(candidatos.margen_px)) if candidatos.margen_px.size else None
            ),
            "fotogramas": candidatos.fotogramas.tolist(),
        }
    return banderas, tasas


@dataclass(frozen=True)
class ResultadoFiltrado:
    """Qué produjo una corrida de filtrado."""

    ruta_filtrado: Path
    ruta_metadata: Path
    muestras: int
    muestras_con_dato: int
    muestras_interpoladas: int
    muestras_filtradas: int
    fotogramas_interpolados: int
    tramos_filtrados: int
    tramos_demasiado_cortos: int
    intercambios: dict[str, dict[str, Any]]


def filtrar_corrida(
    corrida: str | Path, destino: str | Path, configuracion: Configuracion
) -> ResultadoFiltrado:
    """Filtra los landmarks de una corrida de extracción y los persiste."""
    corrida = Path(corrida)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    tipo = configuracion.exigir("filtrado.tipo")
    if tipo not in TIPOS_SOPORTADOS:
        raise ErrorDeConfiguracion(
            f"filtrado.tipo '{tipo}' no está implementado; "
            f"soportados: {', '.join(TIPOS_SOPORTADOS)}"
        )
    orden = configuracion.exigir("filtrado.orden")
    corte = configuracion.exigir("filtrado.frecuencia_corte_hz")
    hueco_maximo = configuracion.exigir("filtrado.hueco_maximo_interpolable_fotogramas")
    margen = configuracion.exigir("lateralidad.margen_minimo_px")
    separacion = configuracion.exigir("lateralidad.separacion_minima_px")

    series = cargar_series(corrida)
    fotogramas = series.cantidad_fotogramas
    minimo = longitud_minima_para_filtfilt(orden)

    x = np.empty_like(series.x)
    y = np.empty_like(series.y)
    interpolado = np.zeros(series.x.shape, dtype=bool)
    filtrado = np.zeros(series.x.shape, dtype=bool)
    tramos_filtrados = 0
    tramos_cortos = 0

    for landmark in range(CANTIDAD_LANDMARKS):
        for salida, cruda in ((x, series.x), (y, series.y)):
            rellena, marca_interpolado = interpolar_huecos_cortos(cruda[:, landmark], hueco_maximo)
            suave, marca_filtrado = filtrar_trayectoria(rellena, series.fps, orden, corte)
            salida[:, landmark] = suave
            interpolado[:, landmark] |= marca_interpolado
            filtrado[:, landmark] |= marca_filtrado
        # Los tramos se cuentan una sola vez por landmark: x e y tienen los
        # mismos huecos, porque un fotograma sin detección no trae ninguna de las dos.
        for inicio, fin in tramos_continuos(~np.isnan(x[:, landmark])):
            if fin - inicio >= minimo:
                tramos_filtrados += 1
            else:
                tramos_cortos += 1

    banderas, intercambios = marcar_intercambios(series, margen, separacion)

    tabla = _tabla_filtrada(series, x, y, interpolado, filtrado, banderas)
    ruta_filtrado = destino / NOMBRE_FILTRADO
    pq.write_table(tabla, ruta_filtrado, compression=COMPRESION)

    con_dato = int((~np.isnan(x)).sum())
    resultado = ResultadoFiltrado(
        ruta_filtrado=ruta_filtrado,
        ruta_metadata=destino / NOMBRE_METADATA,
        muestras=fotogramas * CANTIDAD_LANDMARKS,
        muestras_con_dato=con_dato,
        muestras_interpoladas=int(interpolado.sum()),
        muestras_filtradas=int(filtrado.sum()),
        fotogramas_interpolados=int(interpolado.any(axis=1).sum()),
        tramos_filtrados=tramos_filtrados,
        tramos_demasiado_cortos=tramos_cortos,
        intercambios=intercambios,
    )
    escribir_metadata(
        _metadata(corrida, configuracion, series, resultado, orden, corte, hueco_maximo, minimo),
        resultado.ruta_metadata,
    )
    return resultado


def _tabla_filtrada(
    series: SeriesDeLandmarks,
    x: np.ndarray,
    y: np.ndarray,
    interpolado: np.ndarray,
    filtrado: np.ndarray,
    banderas: np.ndarray,
) -> pa.Table:
    fotogramas, landmarks = x.shape
    frame = np.repeat(np.arange(fotogramas, dtype=np.int32), landmarks)
    timestamp = np.repeat(series.timestamp_ms, landmarks)
    landmark_id = np.tile(np.arange(landmarks, dtype=np.int16), fotogramas)
    return pa.Table.from_arrays(
        [
            pa.array(frame),
            pa.array(timestamp),
            pa.array(landmark_id),
            pa.array(x.ravel().astype(np.float32)),
            pa.array(y.ravel().astype(np.float32)),
            pa.array(series.visibility.ravel().astype(np.float32)),
            pa.array(interpolado.ravel()),
            pa.array(filtrado.ravel()),
            pa.array(banderas.ravel()),
        ],
        schema=ESQUEMA_FILTRADO,
    )


def _metadata(
    corrida: Path,
    configuracion: Configuracion,
    series: SeriesDeLandmarks,
    resultado: ResultadoFiltrado,
    orden: int,
    corte: float,
    hueco_maximo: int,
    minimo: int,
) -> dict[str, Any]:
    origen = corrida / "landmarks.parquet"
    metadata_origen = corrida / "metadata.json"
    return {
        **encabezado_de_corrida("filtrado"),
        # La metadata de la extracción va entera acá adentro: con este solo
        # archivo se reconstruye la cadena hasta el video.
        "origen": {
            "corrida": str(corrida),
            "landmarks": str(origen),
            "sha256_landmarks": sha256_de_archivo(origen) if origen.is_file() else None,
            "metadata_extraccion": (
                json.loads(metadata_origen.read_text(encoding="utf-8"))
                if metadata_origen.is_file()
                else None
            ),
        },
        "parametros": {
            "tipo": configuracion.filtrado.tipo,
            "orden": orden,
            "frecuencia_corte_hz": corte,
            "hueco_maximo_interpolable_fotogramas": hueco_maximo,
            "longitud_minima_de_tramo_filtrable": minimo,
            "gate_por_visibility": False,
            "umbral_visibility_reporte": configuracion.calidad.umbral_visibility_reporte,
            "metodo_correccion_intercambios": (
                configuracion.lateralidad.metodo_correccion_intercambios
            ),
            "margen_minimo_px": configuracion.lateralidad.margen_minimo_px,
            "separacion_minima_px": configuracion.lateralidad.separacion_minima_px,
        },
        "resultado": {
            "archivo": resultado.ruta_filtrado.name,
            "fotogramas": series.cantidad_fotogramas,
            "muestras": resultado.muestras,
            "muestras_con_dato": resultado.muestras_con_dato,
            "muestras_interpoladas": resultado.muestras_interpoladas,
            "muestras_filtradas": resultado.muestras_filtradas,
            "fotogramas_interpolados": resultado.fotogramas_interpolados,
            "tramos_filtrados": resultado.tramos_filtrados,
            "tramos_demasiado_cortos": resultado.tramos_demasiado_cortos,
            "intercambios_por_par": resultado.intercambios,
        },
    }
