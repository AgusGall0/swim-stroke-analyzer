"""Ángulos articulares del lado cercano a la cámara, fotograma a fotograma.

**Qué mide un ángulo articular acá.** Tres landmarks definen dos segmentos
corporales que comparten un vértice: el ángulo es el que forman esos segmentos
*en el vértice*, medido entre 0° y 180°. Es el ángulo incluido, no el ángulo de
flexión de la convención anatómica: 180° es el segmento extendido (los tres
puntos alineados) y el valor baja a medida que la articulación se cierra. Se
elige el ángulo incluido porque es la medición directa, sin restarle nada a
nada: cualquier convención de flexión se deriva después sin volver al video.

Las tres articulaciones del lado izquierdo, que es el cercano a la cámara:

- **Codo** (hombro-codo-muñeca): el ángulo entre brazo y antebrazo. En crol
  arranca cerca de 180° en la entrada y extensión, baja durante el agarre
  (*catch*) y el tirón (*pull*) —el codo alto de la técnica es justamente ese
  cierre—, vuelve a abrirse al final del empuje (*push*) y se cierra de nuevo
  en el recobro (*recovery*). Es la señal que mejor describe el ciclo.
- **Hombro** (codo-hombro-cadera): el ángulo entre el brazo y el tronco. Cerca
  de 0° con el brazo pegado al cuerpo, cerca de 180° con el brazo estirado
  adelante en línea con el tronco. Describe dónde está el brazo respecto del
  cuerpo, que es lo que el ángulo de codo por sí solo no dice.
- **Rodilla** (cadera-rodilla-tobillo): 180° con la pierna extendida. Entra
  para el batido de piernas, aunque en este material los landmarks de la pierna
  son los menos confiables.

**Se calcula sobre coordenadas en píxeles.** El Parquet filtrado ya las trae
convertidas (``x_px``, ``y_px``). Calcular el ángulo sobre coordenadas
normalizadas lo distorsionaría por la relación de aspecto: 576×324 no es
cuadrado, y un mismo ángulo real daría distinto según su orientación.

**Es un ángulo proyectado, no el ángulo real.** La medición vive en el plano de
la imagen. Cuando el segmento sale del plano —y en crol sale, porque el cuerpo
rota sobre su eje— el ángulo proyectado difiere del real, y puede quedar tanto
por debajo como por encima: dos segmentos casi alineados con el eje de la cámara
proyectan a casi 180°. Un valor bajo puede ser flexión o puede ser escorzo, y
con una sola cámara no se distinguen.

**Un ángulo es tan confiable como su peor componente.** Cada ángulo hereda las
banderas de sus tres landmarks: si a alguno le faltó pasar por el filtro, se
rellenó por interpolación, quedó sospechado de intercambio izquierda/derecha o
tiene visibility por debajo del umbral de reporte, el ángulo queda marcado con
ese motivo. Un landmark sin visibility cuenta como visibility baja: no tenerla
no es tenerla alta. Marcar no es descartar: el valor se calcula y se guarda igual, y
quien lo use decide. Los fotogramas sin las tres mediciones no dan ángulo: la
fila se escribe con ``NaN`` y el hueco queda explícito.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from swimalyzer.config import Configuracion, ErrorDeConfiguracion
from swimalyzer.io.metadata import encabezado_de_corrida, escribir_metadata, sha256_de_archivo
from swimalyzer.io.persistencia import COMPRESION
from swimalyzer.pose import landmarks as lm
from swimalyzer.signal.filtrado import (
    NOMBRE_FILTRADO,
    NOMBRE_METADATA,
    SeriesFiltradas,
    cargar_filtrado,
)

NOMBRE_ANGULOS = "angulos.parquet"
NOMBRE_METADATA_ANGULOS = "metadata_angulos.json"


@dataclass(frozen=True)
class Articulacion:
    """Qué tres landmarks definen un ángulo y cuál es el vértice.

    ``proximal`` y ``distal`` son los extremos de los dos segmentos; el ángulo
    se mide en ``vertice``, que es la articulación propiamente dicha.
    """

    nombre: str
    #: Tipo de articulación ("codo", "hombro", "rodilla"), sin el lado: es la
    #: clave del rango anatómico, que no depende de si es la izquierda o la
    #: derecha.
    tipo: str
    proximal: int
    vertice: int
    distal: int

    @property
    def landmarks(self) -> tuple[int, int, int]:
        return (self.proximal, self.vertice, self.distal)

    @property
    def descripcion(self) -> str:
        return "-".join(lm.nombre(landmark) for landmark in self.landmarks)


#: Las articulaciones del lado izquierdo, el cercano a la cámara. El lado
#: derecho no se calcula: con este material su codo no tiene medición usable en
#: el 96 % de los fotogramas (ver el informe de caracterización).
ARTICULACIONES: tuple[Articulacion, ...] = (
    Articulacion("codo_izq", "codo", lm.HOMBRO_IZQ, lm.CODO_IZQ, lm.MUNECA_IZQ),
    Articulacion("hombro_izq", "hombro", lm.CODO_IZQ, lm.HOMBRO_IZQ, lm.CADERA_IZQ),
    Articulacion("rodilla_izq", "rodilla", lm.CADERA_IZQ, lm.RODILLA_IZQ, lm.TOBILLO_IZQ),
)

#: Por qué puede quedar marcado un ángulo. El orden es el del informe.
MOTIVOS: tuple[str, ...] = (
    "sin_filtrar",
    "interpolado",
    "intercambio_sospechado",
    "visibility_baja",
    "fuera_de_rango_en_el_plano_medido",
    "velocidad_angular",
)

ESQUEMA_ANGULOS = pa.schema(
    [
        pa.field("frame", pa.int32(), nullable=False),
        pa.field("timestamp_ms", pa.int64(), nullable=False),
        pa.field("articulacion", pa.string(), nullable=False),
        # En grados, sobre coordenadas en píxeles. NaN si falta alguno de los
        # tres landmarks: el fotograma sin dato se registra, no se saltea.
        pa.field("angulo_grados", pa.float32()),
        # La peor visibility de los tres landmarks: el ángulo no puede ser más
        # confiable que eso.
        pa.field("visibility_minima", pa.float32()),
        pa.field("sin_filtrar", pa.bool_(), nullable=False),
        pa.field("interpolado", pa.bool_(), nullable=False),
        pa.field("intercambio_sospechado", pa.bool_(), nullable=False),
        pa.field("visibility_baja", pa.bool_(), nullable=False),
        pa.field("fuera_de_rango_en_el_plano_medido", pa.bool_(), nullable=False),
        pa.field("velocidad_angular", pa.bool_(), nullable=False),
        pa.field("marcado", pa.bool_(), nullable=False),
    ]
)


def angulo_interior(proximal: np.ndarray, vertice: np.ndarray, distal: np.ndarray) -> np.ndarray:
    """Ángulo en grados que forman en ``vertice`` los segmentos hacia los extremos.

    Cada argumento es un arreglo ``(n, 2)`` de coordenadas **en píxeles**.

    Se usa ``atan2(|producto vectorial|, producto escalar)`` y no
    ``arccos`` del coseno: el coseno cerca de 0° y de 180° cambia muy poco para
    variaciones grandes del ángulo, así que ahí ``arccos`` amplifica el error
    numérico y además se sale del dominio por redondeo. ``atan2`` no tiene esos
    dos problemas y devuelve directo el rango 0 a 180.

    Un segmento de longitud cero no define dirección: ese ángulo sale ``NaN``.
    """
    u = np.asarray(proximal, dtype=np.float64) - np.asarray(vertice, dtype=np.float64)
    v = np.asarray(distal, dtype=np.float64) - np.asarray(vertice, dtype=np.float64)
    producto_vectorial = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    producto_escalar = (u * v).sum(axis=-1)
    grados = np.degrees(np.arctan2(np.abs(producto_vectorial), producto_escalar))
    degenerado = np.isclose(np.linalg.norm(u, axis=-1), 0) | np.isclose(
        np.linalg.norm(v, axis=-1), 0
    )
    return np.where(degenerado, np.nan, grados)


@dataclass(frozen=True)
class SerieDeAngulo:
    """La serie temporal de una articulación, con sus banderas por fotograma."""

    articulacion: Articulacion
    grados: np.ndarray
    visibility_minima: np.ndarray
    #: ``{motivo: máscara de fotogramas}``, una entrada por cada uno de MOTIVOS.
    motivos: dict[str, np.ndarray]

    @property
    def con_dato(self) -> np.ndarray:
        return ~np.isnan(self.grados)

    @property
    def marcado(self) -> np.ndarray:
        """Marcado por cualquier motivo. Solo aplica donde hay ángulo."""
        marcas = np.zeros(self.grados.size, dtype=bool)
        for mascara in self.motivos.values():
            marcas |= mascara
        return marcas & self.con_dato

    @property
    def limpio(self) -> np.ndarray:
        """Fotogramas con ángulo y sin ninguna bandera."""
        return self.con_dato & ~self.marcado


def calcular_angulo(
    series: SeriesFiltradas,
    articulacion: Articulacion,
    umbral_visibility: float,
    rango_anatomico: tuple[float, float],
    velocidad_maxima: float,
) -> SerieDeAngulo:
    """Calcula una articulación sobre toda la corrida y la marca con sus criterios.

    Cuatro de los seis motivos se heredan de los landmarks. Los otros dos miran
    el ángulo ya calculado, porque sobre este material las banderas heredadas
    resultaron necesarias pero no suficientes: la ``visibility`` de MediaPipe no
    predice si el ángulo derivado tiene sentido.
    """
    puntos = [
        np.stack([series.x[:, landmark], series.y[:, landmark]], axis=1)
        for landmark in articulacion.landmarks
    ]
    grados = angulo_interior(*puntos)

    indices = list(articulacion.landmarks)
    visibility = series.visibility[:, indices]
    sin_visibility = np.isnan(visibility)
    # El mínimo se calcula con los NaN puestos en +inf y se devuelven a NaN
    # después: un landmark sin visibility no es un landmark con visibility alta.
    visibility_minima = np.where(sin_visibility, np.inf, visibility).min(axis=1)
    visibility_minima = np.where(np.isinf(visibility_minima), np.nan, visibility_minima)
    with np.errstate(invalid="ignore"):
        baja = np.any(sin_visibility | (visibility < umbral_visibility), axis=1)
    minimo, maximo = rango_anatomico
    with np.errstate(invalid="ignore"):
        fuera_de_rango = (grados < minimo) | (grados > maximo)
        rapido = salto_adyacente(velocidad_angular(grados)) >= velocidad_maxima
    motivos = {
        # "sin filtrar" es la ausencia de la bandera: el tramo era más corto
        # que el mínimo de filtfilt y quedó con la señal cruda.
        "sin_filtrar": ~np.all(series.filtrado[:, indices], axis=1),
        "interpolado": np.any(series.interpolado[:, indices], axis=1),
        "intercambio_sospechado": np.any(series.intercambio_sospechado[:, indices], axis=1),
        "visibility_baja": baja,
        # El nombre dice exactamente lo que se sabe. Un valor fuera del rango
        # que la articulación puede recorrer puede venir de un landmark mal
        # estimado o de escorzo extremo —con el segmento apuntando a la cámara,
        # la proyección puede achicar el ángulo tanto como agrandarlo—, y con
        # una sola cámara no se distingue cuál de los dos es. En ambos casos la
        # medición no representa a la articulación, que es lo que justifica
        # marcarla; culpar al landmark sería afirmar más de lo que se sabe.
        "fuera_de_rango_en_el_plano_medido": fuera_de_rango,
        # Un salto que el cuerpo no puede hacer en 1/fps de segundo, mirando el
        # que entra y el que sale para señalar el fotograma y no la transición.
        "velocidad_angular": rapido,
    }
    sin_dato = np.isnan(grados)
    # Donde no hay ángulo no hay nada que marcar: el hueco ya se informa aparte.
    motivos = {motivo: mascara & ~sin_dato for motivo, mascara in motivos.items()}
    return SerieDeAngulo(
        articulacion=articulacion,
        grados=grados,
        visibility_minima=np.where(sin_dato, np.nan, visibility_minima),
        motivos=motivos,
    )


def velocidad_angular(grados: np.ndarray) -> np.ndarray:
    """Cuánto cambió el ángulo respecto del fotograma anterior, en grados por fotograma.

    Se devuelve el módulo: para decidir si un cambio es creíble no importa si la
    articulación se abrió o se cerró, importa cuánto se movió en 1/fps de
    segundo. La unidad es grados **por fotograma** y no por segundo porque el
    dato es discreto: entre dos fotogramas no hay nada, y una velocidad
    instantánea ahí es una interpolación que el video no respalda.

    El primer fotograma y los que siguen a un hueco quedan en ``NaN``: no tienen
    con qué compararse. Un salto a través de un hueco no es velocidad, es la
    suma de todo lo que pasó mientras no hubo medición.
    """
    salida = np.full(grados.size, np.nan, dtype=np.float64)
    salida[1:] = np.abs(np.diff(grados))
    return salida


def salto_adyacente(velocidad: np.ndarray) -> np.ndarray:
    """El mayor de los dos saltos que tocan cada fotograma: el que entra y el que sale.

    Un valor aislado y disparatado produce dos saltos grandes, uno para llegar y
    otro para volver. Mirar los dos es lo que permite señalar **el fotograma**
    sospechoso y no solo la transición.
    """
    entra = velocidad
    sale = np.r_[velocidad[1:], np.nan]
    with np.errstate(invalid="ignore"):
        return np.fmax(entra, sale)


def rango(grados: np.ndarray) -> dict[str, Any]:
    """Mínimo, percentiles y máximo de una serie de ángulos, ignorando huecos."""
    valores = grados[~np.isnan(grados)]
    if valores.size == 0:
        return {"n": 0}
    p5, p50, p95 = np.percentile(valores, (5, 50, 95))
    return {
        "n": int(valores.size),
        "minimo": float(valores.min()),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "maximo": float(valores.max()),
    }


@dataclass(frozen=True)
class ResumenDeArticulacion:
    """Cobertura, marcado y rango de una articulación en una corrida."""

    nombre: str
    fotogramas: int
    con_dato: int
    marcados: int
    por_motivo: dict[str, int]
    rango: dict[str, Any]
    rango_sin_marcar: dict[str, Any]

    @property
    def tasa_con_dato(self) -> float:
        return self.con_dato / self.fotogramas if self.fotogramas else 0.0

    @property
    def tasa_marcados(self) -> float:
        """Sobre los ángulos calculados, que es lo que el marcado califica."""
        return self.marcados / self.con_dato if self.con_dato else 0.0


def resumir(serie: SerieDeAngulo) -> ResumenDeArticulacion:
    """Cuántos ángulos hay, cuántos quedan marcados y por qué motivo."""
    return ResumenDeArticulacion(
        nombre=serie.articulacion.nombre,
        fotogramas=int(serie.grados.size),
        con_dato=int(serie.con_dato.sum()),
        marcados=int(serie.marcado.sum()),
        # Los motivos no son excluyentes: un ángulo puede estar marcado por
        # varios y la suma de la columna puede pasar el total de marcados.
        por_motivo={motivo: int(mascara.sum()) for motivo, mascara in serie.motivos.items()},
        rango=rango(serie.grados),
        rango_sin_marcar=rango(np.where(serie.limpio, serie.grados, np.nan)),
    )


@dataclass(frozen=True)
class ResultadoAngulos:
    """Qué produjo una corrida de cálculo de ángulos."""

    ruta_angulos: Path
    ruta_metadata: Path
    fotogramas: int
    fps: float
    umbral_visibility: float
    rangos_anatomicos: dict[str, tuple[float, float]]
    velocidad_maxima: float
    series: dict[str, SerieDeAngulo]
    resumenes: dict[str, ResumenDeArticulacion]


def _rangos_anatomicos(configuracion: Configuracion) -> dict[str, tuple[float, float]]:
    """El rango de cada articulación, buscado por tipo y devuelto por nombre.

    La configuración los guarda por tipo ("codo") porque el rango de movimiento
    no depende del lado. Si falta el tipo de alguna articulación que se calcula,
    es un error de configuración y no un valor por defecto: inventar un rango
    permisivo apagaría el criterio en silencio.
    """
    por_tipo = configuracion.exigir("calidad.rango_anatomico_grados")
    rangos: dict[str, tuple[float, float]] = {}
    for articulacion in ARTICULACIONES:
        rango_del_tipo = por_tipo.get(articulacion.tipo)
        if rango_del_tipo is None:
            raise ErrorDeConfiguracion(
                f"falta el rango anatómico de '{articulacion.tipo}' en "
                f"calidad.rango_anatomico_grados, que hace falta para "
                f"'{articulacion.nombre}'"
            )
        rangos[articulacion.nombre] = (rango_del_tipo.minimo, rango_del_tipo.maximo)
    return rangos


def calcular_angulos_de_corrida(
    corrida: str | Path, destino: str | Path, configuracion: Configuracion
) -> ResultadoAngulos:
    """Calcula los ángulos de una corrida ya filtrada y los persiste en Parquet."""
    corrida = Path(corrida)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    umbral = configuracion.exigir("calidad.umbral_visibility_reporte")
    rangos = _rangos_anatomicos(configuracion)
    velocidad_maxima = configuracion.exigir("calidad.velocidad_angular_maxima_grados_por_fotograma")

    series = cargar_filtrado(corrida)
    calculadas = {
        articulacion.nombre: calcular_angulo(
            series, articulacion, umbral, rangos[articulacion.nombre], velocidad_maxima
        )
        for articulacion in ARTICULACIONES
    }

    ruta_angulos = destino / NOMBRE_ANGULOS
    pq.write_table(_tabla(series, calculadas), ruta_angulos, compression=COMPRESION)

    resultado = ResultadoAngulos(
        ruta_angulos=ruta_angulos,
        ruta_metadata=destino / NOMBRE_METADATA_ANGULOS,
        fotogramas=series.cantidad_fotogramas,
        fps=series.fps,
        umbral_visibility=umbral,
        rangos_anatomicos=rangos,
        velocidad_maxima=velocidad_maxima,
        series=calculadas,
        resumenes={nombre: resumir(serie) for nombre, serie in calculadas.items()},
    )
    escribir_metadata(_metadata(corrida, configuracion, resultado), resultado.ruta_metadata)
    return resultado


def cargar_angulos(directorio: str | Path) -> pa.Table:
    """Lee un Parquet de ángulos y verifica que tenga el esquema esperado."""
    archivo = Path(directorio) / NOMBRE_ANGULOS
    tabla = pq.read_table(archivo)
    faltantes = [campo.name for campo in ESQUEMA_ANGULOS if campo.name not in tabla.column_names]
    if faltantes:
        raise ValueError(f"{archivo} no es un Parquet de ángulos: faltan {faltantes}")
    return tabla


def cargar_series_de_angulo(directorio: str | Path) -> dict[str, SerieDeAngulo]:
    """Reconstruye las series de ángulos desde el Parquet ya persistido.

    Es la puerta de entrada de las figuras y los informes: dibujan lo que está
    guardado, no un cálculo hecho al paso que nadie podría volver a revisar.
    """
    tabla = cargar_angulos(directorio)
    datos = {nombre: np.asarray(tabla.column(nombre)) for nombre in tabla.column_names}
    por_nombre = {articulacion.nombre: articulacion for articulacion in ARTICULACIONES}

    series: dict[str, SerieDeAngulo] = {}
    for nombre in dict.fromkeys(datos["articulacion"].tolist()):
        filas = np.where(datos["articulacion"] == nombre)[0]
        # El Parquet no promete orden: las filas se reordenan por fotograma
        # antes de armar la serie.
        filas = filas[np.argsort(datos["frame"][filas], kind="stable")]
        series[nombre] = SerieDeAngulo(
            articulacion=por_nombre[nombre],
            grados=datos["angulo_grados"][filas].astype(np.float64),
            visibility_minima=datos["visibility_minima"][filas].astype(np.float64),
            motivos={motivo: datos[motivo][filas].astype(bool) for motivo in MOTIVOS},
        )
    return series


def _tabla(series: SeriesFiltradas, calculadas: dict[str, SerieDeAngulo]) -> pa.Table:
    fotogramas = series.cantidad_fotogramas
    frame = np.tile(np.arange(fotogramas, dtype=np.int32), len(calculadas))
    timestamp = np.tile(series.timestamp_ms, len(calculadas))
    nombres = np.repeat(list(calculadas), fotogramas)
    apiladas = list(calculadas.values())

    def columna(valores: list[np.ndarray], tipo: Any) -> pa.Array:
        return pa.array(np.concatenate(valores).astype(tipo))

    return pa.Table.from_arrays(
        [
            pa.array(frame),
            pa.array(timestamp),
            pa.array(nombres),
            columna([serie.grados for serie in apiladas], np.float32),
            columna([serie.visibility_minima for serie in apiladas], np.float32),
            *(columna([serie.motivos[motivo] for serie in apiladas], bool) for motivo in MOTIVOS),
            columna([serie.marcado for serie in apiladas], bool),
        ],
        schema=ESQUEMA_ANGULOS,
    )


def _metadata(
    corrida: Path, configuracion: Configuracion, resultado: ResultadoAngulos
) -> dict[str, Any]:
    origen = corrida / NOMBRE_FILTRADO
    metadata_origen = corrida / NOMBRE_METADATA
    return {
        **encabezado_de_corrida("angulos"),
        "origen": {
            "corrida": str(corrida),
            "landmarks_filtrados": str(origen),
            "sha256_landmarks_filtrados": sha256_de_archivo(origen) if origen.is_file() else None,
            "metadata_filtrado": (
                json.loads(metadata_origen.read_text(encoding="utf-8"))
                if metadata_origen.is_file()
                else None
            ),
        },
        "parametros": {
            "umbral_visibility_reporte": resultado.umbral_visibility,
            "rango_anatomico_grados": {
                nombre: {"minimo": minimo, "maximo": maximo}
                for nombre, (minimo, maximo) in resultado.rangos_anatomicos.items()
            },
            "velocidad_angular_maxima_grados_por_fotograma": resultado.velocidad_maxima,
            "lado": "izquierdo (cercano a la cámara)",
            "coordenadas": "pixeles",
            "convencion": "angulo incluido en el vertice, 0 a 180 grados; 180 = extendido",
            "articulaciones": {
                articulacion.nombre: articulacion.descripcion for articulacion in ARTICULACIONES
            },
        },
        "resultado": {
            "archivo": resultado.ruta_angulos.name,
            "fotogramas": resultado.fotogramas,
            "fps": resultado.fps,
            "por_articulacion": {
                nombre: {
                    "con_dato": resumen.con_dato,
                    "sin_dato": resumen.fotogramas - resumen.con_dato,
                    "marcados": resumen.marcados,
                    "tasa_marcados_sobre_calculados": resumen.tasa_marcados,
                    "por_motivo": resumen.por_motivo,
                    "rango_grados": resumen.rango,
                    "rango_grados_sin_marcar": resumen.rango_sin_marcar,
                }
                for nombre, resumen in resultado.resumenes.items()
            },
        },
    }
