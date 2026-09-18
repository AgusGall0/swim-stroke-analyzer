"""Segmentación de la serie en ciclos de brazada y curva media normalizada.

**Qué es un ciclo de brazada acá.** Un ciclo de un brazo en crol es el intervalo
entre dos repeticiones del mismo instante del movimiento. Cualquier instante
sirve como corte mientras sea el mismo en todos los ciclos; lo que hace que un
corte sea bueno es que corresponda a un evento identificable de la brazada y no
a un accidente de la señal, porque de eso depende que el 0 % de un ciclo sea
comparable con el 0 % del siguiente.

**El evento de corte es la mano en lo más alto del recobro (*recovery*), justo
antes de la entrada al agua.** Es el punto de referencia habitual en la
literatura para delimitar el ciclo, y en esta serie se mide como el máximo de la
altura de la muñeca sobre el hombro. Se corta ahí y no en un extremo del ángulo
de codo por evidencia: sobre este material el ángulo de codo vive saturado entre
165° y 180° y el detector de picos engancha la meseta, con lo que da ciclos de
1.40 a 3.07 s. Las señales de posición, en cambio, dan ciclos parejos.

**La señal de corte es vertical y relativa al hombro.** Relativa porque el
nadador se desplaza dentro del encuadre y una coordenada absoluta mezclaría ese
desplazamiento con el movimiento del brazo. Vertical porque es la componente que
separa el recobro —brazo arriba— del resto del ciclo.

**Ojo con el signo.** En coordenadas de imagen la ``y`` crece hacia abajo, así
que la altura de la muñeca sobre el hombro es ``y_hombro - y_muñeca``: positiva
cuando la mano está por encima del hombro. Tomar ``y_muñeca - y_hombro`` y
buscarle el máximo cortaría en la mano más profunda del tirón (*pull*), que es
otro evento y da otros ciclos.

**Un ciclo no cruza un hueco.** Los picos se buscan tramo continuo por tramo
continuo, y un ciclo vive entero dentro de un tramo. Un intervalo que abarque un
hueco no es un ciclo, es dos trozos con un tiempo indeterminado en el medio.

**Normalizar al 0-100 % del ciclo** permite promediar ciclos de distinta
duración: cada uno se reinterpola linealmente sobre la misma malla de
porcentajes, y ahí el promedio compara fases equivalentes en vez de fotogramas
equivalentes. Es la representación canónica en biomecánica.

**Los ciclos con mediciones marcadas entran igual.** Con siete ciclos,
descartar los que tienen alguna medición marcada deja la muestra en nada y, peor,
esconde el problema: la figura saldría limpia porque se le sacó lo sucio, no
porque el dato lo sea. En vez de descartar, la cobertura viaja con la curva: en
cada punto del ciclo se informa qué fracción de las mediciones que se promediaron
ahí está marcada por algún motivo.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import find_peaks

from swimalyzer.config import Configuracion, ErrorDeConfiguracion
from swimalyzer.io.metadata import encabezado_de_corrida, escribir_metadata, sha256_de_archivo
from swimalyzer.io.persistencia import COMPRESION
from swimalyzer.metrics.angulos import (
    NOMBRE_ANGULOS,
    NOMBRE_METADATA_ANGULOS,
    SerieDeAngulo,
    cargar_series_de_angulo,
)
from swimalyzer.pose import landmarks as lm
from swimalyzer.signal.caracterizacion import tramos_continuos
from swimalyzer.signal.filtrado import SeriesFiltradas, cargar_filtrado

NOMBRE_CICLOS = "ciclos.parquet"
NOMBRE_METADATA_CICLOS = "metadata_ciclos.json"

#: Articulación cuya curva media es el entregable. La segmentación se aplica a
#: todas, pero la figura de referencia es la de codo: es la que describe el
#: ciclo.
ARTICULACION_DE_REFERENCIA = "codo_izq"

#: Puntos de la malla de porcentaje del ciclo: 0, 1, ..., 100. Uno por punto
#: porcentual, que es la resolución con la que se lee una curva de ciclo; más
#: puntos no agregan información porque los ciclos duran unos 55 fotogramas.
MALLA_PORCENTAJE = np.linspace(0.0, 100.0, 101)


class ErrorDeSegmentacion(Exception):
    """La serie no se puede segmentar con los parámetros pedidos."""


@dataclass(frozen=True)
class SenalDeSegmentacion:
    """Una señal candidata para cortar el ciclo, y qué evento es su máximo.

    ``construir`` devuelve la señal en píxeles a partir de las trayectorias
    filtradas. ``evento`` nombra, en términos de la brazada, qué instante es el
    máximo de esa señal: es lo que define la fase 0 del ciclo.
    """

    nombre: str
    descripcion: str
    evento: str
    construir: Callable[[SeriesFiltradas], np.ndarray]


def _altura_de_muneca_sobre_hombro(series: SeriesFiltradas) -> np.ndarray:
    """Cuántos píxeles está la muñeca izquierda por encima del hombro izquierdo.

    ``y_hombro - y_muñeca`` y no al revés: en coordenadas de imagen la ``y``
    crece hacia abajo, así que restar en este orden deja el valor positivo
    cuando la mano está arriba, que es lo que hace que el máximo sea el punto
    más alto del recobro.
    """
    return series.y[:, lm.HOMBRO_IZQ] - series.y[:, lm.MUNECA_IZQ]


#: Vocabulario de ``segmentacion.senal``. Hay una sola entrada porque la
#: decisión está tomada con evidencia (ver ``config.yaml``); las candidatas que
#: se midieron y se descartaron no se dejan acá como opción, porque ofrecerlas
#: sin su justificación invita a cambiarlas sin ella.
SENALES_DE_SEGMENTACION: dict[str, SenalDeSegmentacion] = {
    "muneca_y_rel_hombro": SenalDeSegmentacion(
        nombre="muneca_y_rel_hombro",
        descripcion="altura de la muñeca izquierda sobre el hombro izquierdo (px, + = arriba)",
        evento="mano en lo más alto del recobro, antes de la entrada al agua",
        construir=_altura_de_muneca_sobre_hombro,
    ),
}

ESQUEMA_CICLOS = pa.schema(
    [
        # Numeración global de los ciclos, en orden temporal.
        pa.field("ciclo", pa.int32(), nullable=False),
        # A qué tramo continuo pertenece: un ciclo no cruza un hueco, así que
        # saber de qué tramo salió cada uno es parte de poder leer la muestra.
        pa.field("tramo", pa.int32(), nullable=False),
        pa.field("frame_inicio", pa.int32(), nullable=False),
        pa.field("frame_fin", pa.int32(), nullable=False),
        pa.field("articulacion", pa.string(), nullable=False),
        # Fase del ciclo, de 0 a 100. 0 % y 100 % son el mismo evento en dos
        # repeticiones consecutivas.
        pa.field("porcentaje_ciclo", pa.float32(), nullable=False),
        # El ángulo reinterpolado a esa fase, en grados.
        pa.field("angulo_grados", pa.float32()),
        # Si la medición más cercana a esa fase estaba marcada. Se propaga por
        # vecino más cercano y no por interpolación: una bandera es booleana y
        # promediarla dentro de un ciclo inventaría un valor intermedio.
        pa.field("marcado", pa.bool_(), nullable=False),
    ]
)


@dataclass(frozen=True)
class Ciclo:
    """Un ciclo de brazada: el intervalo entre dos cortes consecutivos.

    ``inicio`` y ``fin`` son fotogramas y los dos pertenecen al ciclo: el
    fotograma ``fin`` es el 100 % y es también el 0 % del ciclo siguiente.
    """

    numero: int
    tramo: int
    inicio: int
    fin: int

    def duracion_s(self, fps: float) -> float:
        # El intervalo dura (fin - inicio) pasos de 1/fps, no uno por fotograma:
        # entre N+1 fotogramas hay N intervalos.
        return (self.fin - self.inicio) / fps


def construir_senal(series: SeriesFiltradas, nombre: str) -> np.ndarray:
    """Arma la señal de segmentación pedida, o falla nombrando las que existen."""
    senal = SENALES_DE_SEGMENTACION.get(nombre)
    if senal is None:
        raise ErrorDeConfiguracion(
            f"segmentacion.senal '{nombre}' no está implementada; "
            f"disponibles: {', '.join(SENALES_DE_SEGMENTACION)}"
        )
    return senal.construir(series)


def detectar_cortes(
    senal: np.ndarray,
    disponible: np.ndarray,
    fps: float,
    distancia_minima_s: float,
) -> list[tuple[int, list[int]]]:
    """Busca los cortes tramo por tramo y devuelve ``(inicio_del_tramo, cortes)``.

    Un corte es un máximo local de la señal separado de los demás por al menos
    ``distancia_minima_s``. La distancia mínima es lo único que restringe la
    detección: sobre este material exigir además una prominencia mínima no
    cambió ningún pico entre 0 y 20 px, así que sería un parámetro inerte.

    Se recorre tramo por tramo porque ``find_peaks`` sobre la serie entera
    trataría los bordes de un hueco como si fueran muestras consecutivas.
    """
    distancia = max(1, int(round(distancia_minima_s * fps)))
    cortes: list[tuple[int, list[int]]] = []
    for inicio, fin in tramos_continuos(disponible):
        posiciones, _ = find_peaks(senal[inicio:fin], distance=distancia)
        cortes.append((inicio, [inicio + int(posicion) for posicion in posiciones]))
    return cortes


def segmentar(
    senal: np.ndarray,
    disponible: np.ndarray,
    fps: float,
    distancia_minima_s: float,
) -> list[Ciclo]:
    """Divide la serie en ciclos entre cortes consecutivos del mismo tramo.

    Los cortes del borde no arman ciclo: entre el comienzo de un tramo y su
    primer corte hay un trozo de ciclo, no un ciclo, y ese trozo no se puede
    normalizar porque no se sabe qué fracción del ciclo es.
    """
    ciclos: list[Ciclo] = []
    for numero_de_tramo, (_, cortes) in enumerate(
        detectar_cortes(senal, disponible, fps, distancia_minima_s)
    ):
        for anterior, siguiente in pairwise(cortes):
            ciclos.append(
                Ciclo(
                    numero=len(ciclos),
                    tramo=numero_de_tramo,
                    inicio=anterior,
                    fin=siguiente,
                )
            )
    return ciclos


def normalizar(valores: np.ndarray, ciclo: Ciclo) -> np.ndarray:
    """Reinterpola un ciclo sobre la malla de porcentaje, linealmente.

    Lineal y no un ajuste de mayor orden: la serie ya viene filtrada, y un
    ajuste suavizaría de nuevo por el camino sin que quede registrado como
    filtrado.
    """
    tramo = valores[ciclo.inicio : ciclo.fin + 1]
    fase = np.linspace(0.0, 100.0, tramo.size)
    return np.interp(MALLA_PORCENTAJE, fase, tramo)


def normalizar_bandera(banderas: np.ndarray, ciclo: Ciclo) -> np.ndarray:
    """Lleva una bandera booleana a la malla por vecino más cercano.

    Interpolar una bandera daría valores entre 0 y 1 que no significan nada a
    nivel de un ciclo: una medición está marcada o no lo está. El promedio entre
    ciclos, que sí es una fracción, se calcula después.
    """
    tramo = banderas[ciclo.inicio : ciclo.fin + 1]
    indices = np.rint(MALLA_PORCENTAJE / 100.0 * (tramo.size - 1)).astype(int)
    return tramo[indices]


@dataclass(frozen=True)
class CurvaMedia:
    """La curva media de una articulación a lo largo del ciclo, con su dispersión.

    ``cobertura_marcada`` es, para cada punto de la malla, la fracción de los
    ciclos promediados cuya medición en esa fase estaba marcada. Es lo que
    reemplaza al descarte: la curva incluye todo y dice de qué está hecha.
    """

    articulacion: str
    malla: np.ndarray
    media: np.ndarray
    desvio: np.ndarray
    #: Los ciclos que entraron, en orden.
    ciclos: tuple[Ciclo, ...]
    #: Cada ciclo normalizado, ``(n_ciclos, malla)``: la figura dibuja los
    #: individuales debajo de la media.
    curvas: np.ndarray
    #: Las banderas de cada ciclo sobre la malla, ``(n_ciclos, malla)``. Se
    #: guardan por ciclo y no solo promediadas porque el Parquet las persiste
    #: una por una: la cobertura se deriva de acá y no al revés.
    marcas: np.ndarray

    @property
    def cobertura_marcada(self) -> np.ndarray:
        """Fracción de los ciclos promediados que está marcada en cada fase."""
        if not self.n:
            return np.zeros(self.malla.size)
        return self.marcas.mean(axis=0)

    @property
    def n(self) -> int:
        return len(self.ciclos)

    @property
    def tramos(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(ciclo.tramo for ciclo in self.ciclos))


def curva_media(serie: SerieDeAngulo, ciclos: list[Ciclo]) -> CurvaMedia:
    """Promedia los ciclos de una articulación, normalizados al 0-100 %.

    Solo entran los ciclos en los que la articulación tiene ángulo en todos sus
    fotogramas: un ciclo con un hueco adentro no se puede normalizar sin
    inventar el trozo que falta. Cuál se cayó y por qué queda en la metadata.
    """
    completos = [ciclo for ciclo in ciclos if serie.con_dato[ciclo.inicio : ciclo.fin + 1].all()]
    curvas = np.array([normalizar(serie.grados, ciclo) for ciclo in completos])
    marcas = np.array([normalizar_bandera(serie.marcado, ciclo) for ciclo in completos])

    if not completos:
        vacio = np.full(MALLA_PORCENTAJE.size, np.nan)
        return CurvaMedia(
            articulacion=serie.articulacion.nombre,
            malla=MALLA_PORCENTAJE,
            media=vacio,
            desvio=vacio,
            ciclos=(),
            curvas=np.empty((0, MALLA_PORCENTAJE.size)),
            marcas=np.empty((0, MALLA_PORCENTAJE.size), dtype=bool),
        )

    return CurvaMedia(
        articulacion=serie.articulacion.nombre,
        malla=MALLA_PORCENTAJE,
        media=curvas.mean(axis=0),
        # ddof=1: es el desvío de una muestra de ciclos, no de la población.
        # Con un solo ciclo no hay dispersión que estimar y queda NaN, que es
        # lo correcto: no es cero.
        desvio=(
            curvas.std(axis=0, ddof=1) if len(completos) > 1 else np.full(curvas.shape[1], np.nan)
        ),
        ciclos=tuple(completos),
        curvas=curvas,
        marcas=marcas,
    )


@dataclass(frozen=True)
class ResultadoCiclos:
    """Qué produjo una corrida de segmentación."""

    ruta_ciclos: Path
    ruta_metadata: Path
    fotogramas: int
    fps: float
    senal: SenalDeSegmentacion
    distancia_minima_s: float
    #: Los tramos continuos ``[inicio, fin)`` sobre los que se buscaron cortes.
    #: Van en el resultado porque el número de tramo de un ciclo no significa
    #: nada sin la lista: el tramo 4 puede no tener ningún ciclo.
    tramos: list[tuple[int, int]]
    ciclos: list[Ciclo]
    curvas: dict[str, CurvaMedia]
    #: La señal de corte y qué fotogramas eran utilizables, para que las figuras
    #: dibujen exactamente lo que se segmentó.
    senal_valores: np.ndarray
    disponible: np.ndarray

    @property
    def duraciones_s(self) -> np.ndarray:
        return np.array([ciclo.duracion_s(self.fps) for ciclo in self.ciclos])

    @property
    def duracion_mediana_s(self) -> float | None:
        """Mediana y no media: con siete ciclos, uno largo —que suele ser uno mal
        cortado— arrastra la media."""
        duraciones = self.duraciones_s
        return float(np.median(duraciones)) if duraciones.size else None

    @property
    def frecuencia_de_brazada_hz(self) -> float | None:
        """La inversa de la duración mediana: la frecuencia de brazada medida."""
        mediana = self.duracion_mediana_s
        return 1.0 / mediana if mediana else None


def segmentar_corrida(
    corrida: str | Path, destino: str | Path, configuracion: Configuracion
) -> ResultadoCiclos:
    """Segmenta en ciclos una corrida con ángulos ya calculados y persiste el resultado."""
    corrida = Path(corrida)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    nombre_senal = configuracion.exigir("segmentacion.senal")
    distancia = configuracion.exigir("segmentacion.distancia_minima_entre_picos_s")
    if nombre_senal not in SENALES_DE_SEGMENTACION:
        raise ErrorDeConfiguracion(
            f"segmentacion.senal '{nombre_senal}' no está implementada; "
            f"disponibles: {', '.join(SENALES_DE_SEGMENTACION)}"
        )
    senal_elegida = SENALES_DE_SEGMENTACION[nombre_senal]

    series = cargar_filtrado(corrida)
    angulos = cargar_series_de_angulo(corrida)
    if ARTICULACION_DE_REFERENCIA not in angulos:
        raise ErrorDeSegmentacion(
            f"el Parquet de ángulos de {corrida} no tiene '{ARTICULACION_DE_REFERENCIA}'"
        )

    senal = construir_senal(series, nombre_senal)
    # Los tramos se definen sobre la señal de corte y el ángulo de referencia a
    # la vez: cortar donde no hay ángulo produciría ciclos que después ninguna
    # curva puede usar.
    disponible = ~np.isnan(senal) & angulos[ARTICULACION_DE_REFERENCIA].con_dato
    ciclos = segmentar(senal, disponible, series.fps, distancia)

    curvas = {nombre: curva_media(serie, ciclos) for nombre, serie in angulos.items()}

    ruta_ciclos = destino / NOMBRE_CICLOS
    pq.write_table(_tabla(curvas), ruta_ciclos, compression=COMPRESION)

    resultado = ResultadoCiclos(
        ruta_ciclos=ruta_ciclos,
        ruta_metadata=destino / NOMBRE_METADATA_CICLOS,
        fotogramas=series.cantidad_fotogramas,
        fps=series.fps,
        senal=senal_elegida,
        distancia_minima_s=distancia,
        tramos=tramos_continuos(disponible),
        ciclos=ciclos,
        curvas=curvas,
        senal_valores=senal,
        disponible=disponible,
    )
    escribir_metadata(_metadata(corrida, resultado), resultado.ruta_metadata)
    return resultado


def _tabla(curvas: dict[str, CurvaMedia]) -> pa.Table:
    """Un renglón por ciclo, articulación y punto de la malla.

    Se guardan los ciclos normalizados y no la media: con los ciclos se puede
    recalcular la media, el desvío y la cobertura, y al revés no.
    """
    columnas: dict[str, list[Any]] = {campo.name: [] for campo in ESQUEMA_CICLOS}
    puntos = MALLA_PORCENTAJE.size
    for nombre, curva in curvas.items():
        for fila, ciclo in enumerate(curva.ciclos):
            columnas["ciclo"] += [ciclo.numero] * puntos
            columnas["tramo"] += [ciclo.tramo] * puntos
            columnas["frame_inicio"] += [ciclo.inicio] * puntos
            columnas["frame_fin"] += [ciclo.fin] * puntos
            columnas["articulacion"] += [nombre] * puntos
            columnas["porcentaje_ciclo"] += MALLA_PORCENTAJE.tolist()
            columnas["angulo_grados"] += curva.curvas[fila].tolist()
            columnas["marcado"] += [bool(valor) for valor in curva.marcas[fila]]
    return pa.Table.from_arrays(
        [pa.array(columnas[campo.name], type=campo.type) for campo in ESQUEMA_CICLOS],
        schema=ESQUEMA_CICLOS,
    )


def _metadata(corrida: Path, resultado: ResultadoCiclos) -> dict[str, Any]:
    origen = corrida / NOMBRE_ANGULOS
    metadata_origen = corrida / NOMBRE_METADATA_ANGULOS
    duraciones = resultado.duraciones_s
    return {
        **encabezado_de_corrida("ciclos"),
        "origen": {
            "corrida": str(corrida),
            "angulos": str(origen),
            "sha256_angulos": sha256_de_archivo(origen) if origen.is_file() else None,
            "metadata_angulos": (
                json.loads(metadata_origen.read_text(encoding="utf-8"))
                if metadata_origen.is_file()
                else None
            ),
        },
        "parametros": {
            "senal": resultado.senal.nombre,
            "senal_descripcion": resultado.senal.descripcion,
            "evento_de_corte": resultado.senal.evento,
            "distancia_minima_entre_picos_s": resultado.distancia_minima_s,
            "prominencia_minima": None,
            "puntos_de_la_malla": int(MALLA_PORCENTAJE.size),
            "ciclos_con_mediciones_marcadas": "se incluyen; la cobertura se informa por fase",
        },
        "resultado": {
            "archivo": resultado.ruta_ciclos.name,
            "fotogramas": resultado.fotogramas,
            "fps": resultado.fps,
            "ciclos_detectados": len(resultado.ciclos),
            # El número de tramo de un ciclo indexa esta lista: no todos los
            # tramos dan ciclos, así que sin ella el número no ubica nada.
            "tramos_continuos": [
                {
                    "numero": numero,
                    "frame_inicio": inicio,
                    "frame_fin": fin - 1,
                    "fotogramas": fin - inicio,
                }
                for numero, (inicio, fin) in enumerate(resultado.tramos)
            ],
            "frecuencia_de_brazada_hz": resultado.frecuencia_de_brazada_hz,
            "duracion_de_ciclo_s": (
                {
                    "minimo": float(duraciones.min()),
                    "mediana": float(np.median(duraciones)),
                    "maximo": float(duraciones.max()),
                }
                if duraciones.size
                else None
            ),
            "ciclos": [
                {
                    "numero": ciclo.numero,
                    "tramo": ciclo.tramo,
                    "frame_inicio": ciclo.inicio,
                    "frame_fin": ciclo.fin,
                    "duracion_s": ciclo.duracion_s(resultado.fps),
                }
                for ciclo in resultado.ciclos
            ],
            "por_articulacion": {
                nombre: {
                    "ciclos_promediados": curva.n,
                    "tramos": list(curva.tramos),
                    "desvio_medio_grados": (
                        float(np.nanmean(curva.desvio)) if curva.n > 1 else None
                    ),
                    "desvio_maximo_grados": (
                        float(np.nanmax(curva.desvio)) if curva.n > 1 else None
                    ),
                    "cobertura_marcada_media": float(curva.cobertura_marcada.mean())
                    if curva.n
                    else None,
                }
                for nombre, curva in resultado.curvas.items()
            },
        },
    }


def cargar_ciclos(directorio: str | Path) -> pa.Table:
    """Lee un Parquet de ciclos y verifica que tenga el esquema esperado."""
    archivo = Path(directorio) / NOMBRE_CICLOS
    tabla = pq.read_table(archivo)
    faltantes = [campo.name for campo in ESQUEMA_CICLOS if campo.name not in tabla.column_names]
    if faltantes:
        raise ValueError(f"{archivo} no es un Parquet de ciclos: faltan {faltantes}")
    return tabla
