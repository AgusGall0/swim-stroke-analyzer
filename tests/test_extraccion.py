"""Extracción de landmarks: fps, recorte, registro de huecos y persistencia."""

import json
import math

import cv2 as cv
import numpy as np
import pytest
import yaml

from swimalyzer.config import Recorte, cargar_configuracion
from swimalyzer.io.persistencia import AcumuladorDeLandmarks, leer_landmarks
from swimalyzer.io.video import (
    ErrorDeVideo,
    LectorDeVideo,
    aplicar_recorte,
    resolver_fps,
    validar_recorte,
)
from swimalyzer.pose.deteccion import MuestraLandmark
from swimalyzer.pose.extraccion import NOMBRE_LANDMARKS, NOMBRE_METADATA, extraer
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS

from .conftest import CONFIG_DEL_REPO

FOTOGRAMAS = 20
ANCHO, ALTO = 64, 48


@pytest.fixture
def video(tmp_path):
    """Video sintético de 20 fotogramas, suficiente para recorrer el pipeline."""
    archivo = tmp_path / "video.avi"
    escritor = cv.VideoWriter(str(archivo), cv.VideoWriter_fourcc(*"MJPG"), 30.0, (ANCHO, ALTO))
    assert escritor.isOpened(), "no se pudo crear el video de prueba"
    for indice in range(FOTOGRAMAS):
        escritor.write(np.full((ALTO, ANCHO, 3), indice * 5, np.uint8))
    escritor.release()
    return archivo


class DetectorFalso:
    """Detector de mentira: no detecta nada en los fotogramas de ``sin_deteccion``."""

    def __init__(self, sin_deteccion=()):
        self.sin_deteccion = set(sin_deteccion)
        self.llamadas = []

    def detectar(self, fotograma_bgr, timestamp_ms):
        indice = len(self.llamadas)
        self.llamadas.append((timestamp_ms, fotograma_bgr.shape))
        if indice in self.sin_deteccion:
            return None
        # Coordenadas con decimales a propósito: tienen que sobrevivir enteras.
        return [
            MuestraLandmark(
                x=0.123456 + id_ * 0.001,
                y=0.654321 - id_ * 0.001,
                z=-0.05,
                visibility=0.9,
                presence=0.8,
            )
            for id_ in range(CANTIDAD_LANDMARKS)
        ]


# --- fps -------------------------------------------------------------------


def test_fps_del_contenedor_se_usa_tal_cual():
    assert resolver_fps(30.0, None) == (30.0, 30.0)


@pytest.mark.parametrize("fps_contenedor", [0.0, -1.0, math.nan, math.inf, 90000.0])
def test_fps_no_usable_sin_forzar_es_error(fps_contenedor):
    # El caso que rompía el prototipo: 1000 * frame / 0 dividía por cero.
    with pytest.raises(ErrorDeVideo, match="--fps"):
        resolver_fps(fps_contenedor, None)


def test_fps_forzado_reemplaza_al_del_contenedor_y_lo_deja_registrado():
    assert resolver_fps(0.0, 25.0) == (25.0, None)
    assert resolver_fps(30.0, 25.0) == (25.0, 30.0)


@pytest.mark.parametrize("fps_forzado", [0.0, -5.0, math.nan])
def test_fps_forzado_invalido_es_error(fps_forzado):
    with pytest.raises(ErrorDeVideo):
        resolver_fps(30.0, fps_forzado)


# --- recorte ---------------------------------------------------------------


def test_aplicar_recorte():
    fotograma = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    recortado = aplicar_recorte(fotograma, Recorte(x=8, y=4, ancho=16, alto=10))
    assert recortado.shape == (10, 16, 3)
    assert np.array_equal(recortado, fotograma[4:14, 8:24])
    assert aplicar_recorte(fotograma, None) is fotograma


def test_recorte_fuera_del_fotograma_es_error():
    with pytest.raises(ErrorDeVideo, match="no entra"):
        validar_recorte(Recorte(x=0, y=0, ancho=100, alto=10), 64, 48)


# --- lectura de video ------------------------------------------------------


def test_lector_informa_propiedades_y_recorre_el_video(video):
    with LectorDeVideo(video) as lector:
        assert lector.info.fps == 30.0
        assert (lector.info.ancho, lector.info.alto) == (ANCHO, ALTO)
        assert lector.info.fps_fue_forzado is False
        indices = [indice for indice, _ in lector]
    assert indices == list(range(FOTOGRAMAS))


def test_lector_aplica_el_recorte_a_cada_fotograma(video):
    recorte = Recorte(x=4, y=2, ancho=32, alto=24)
    with LectorDeVideo(video, recorte=recorte) as lector:
        assert (lector.info.ancho, lector.info.alto) == (32, 24)
        assert (lector.info.ancho_original, lector.info.alto_original) == (ANCHO, ALTO)
        formas = {fotograma.shape for _, fotograma in lector}
    assert formas == {(24, 32, 3)}


def test_timestamps_crecen_y_arrancan_en_cero(video):
    with LectorDeVideo(video) as lector:
        marcas = [lector.timestamp_ms(indice) for indice in range(FOTOGRAMAS)]
    assert marcas[0] == 0
    assert all(isinstance(marca, int) for marca in marcas)
    assert all(b > a for a, b in zip(marcas, marcas[1:], strict=False))


def test_abrir_algo_que_no_es_video_es_error(tmp_path):
    archivo = tmp_path / "vacio.mp4"
    archivo.touch()
    with pytest.raises(ErrorDeVideo, match="no pudo abrir"):
        LectorDeVideo(archivo)


# --- acumulador ------------------------------------------------------------


def test_fotograma_sin_deteccion_deja_constancia_con_nan():
    acumulador = AcumuladorDeLandmarks()
    acumulador.agregar(0, 0, DetectorFalso().detectar(np.zeros((1, 1, 3), np.uint8), 0))
    acumulador.agregar(1, 33, None)
    assert acumulador.fotogramas_sin_deteccion == [1]
    tabla = acumulador.tabla().to_pydict()
    assert len(tabla["frame"]) == 2 * CANTIDAD_LANDMARKS
    # El hueco ocupa su lugar en la serie: mismo paso, sin valores.
    assert tabla["frame"][CANTIDAD_LANDMARKS:] == [1] * CANTIDAD_LANDMARKS
    assert tabla["landmark_id"][CANTIDAD_LANDMARKS:] == list(range(CANTIDAD_LANDMARKS))
    assert all(math.isnan(valor) for valor in tabla["x"][CANTIDAD_LANDMARKS:])
    assert all(math.isnan(valor) for valor in tabla["visibility"][CANTIDAD_LANDMARKS:])


# --- pipeline completo -----------------------------------------------------


@pytest.fixture
def configuracion():
    return cargar_configuracion(CONFIG_DEL_REPO)


def test_extraer_escribe_parquet_y_metadata(video, tmp_path, configuracion):
    detector = DetectorFalso(sin_deteccion={3, 4, 11})
    resultado = extraer(
        video,
        tmp_path / "salida",
        configuracion,
        crear_detector=lambda _: detector,
        hashear_video=False,
    )

    assert resultado.fotogramas_procesados == FOTOGRAMAS
    assert resultado.fotogramas_sin_deteccion == (3, 4, 11)
    assert resultado.fotogramas_con_deteccion == FOTOGRAMAS - 3
    assert resultado.filas == FOTOGRAMAS * CANTIDAD_LANDMARKS
    assert resultado.ruta_landmarks.name == NOMBRE_LANDMARKS
    assert resultado.ruta_metadata.name == NOMBRE_METADATA

    tabla = leer_landmarks(resultado.ruta_landmarks)
    assert tabla.num_rows == FOTOGRAMAS * CANTIDAD_LANDMARKS
    assert tabla.column_names == [
        "frame",
        "timestamp_ms",
        "landmark_id",
        "x",
        "y",
        "z",
        "visibility",
        "presence",
    ]
    datos = tabla.to_pydict()
    assert sorted(set(datos["frame"])) == list(range(FOTOGRAMAS))
    # Los fotogramas sin detección están presentes, con NaN.
    faltantes = {
        frame for frame, x in zip(datos["frame"], datos["x"], strict=True) if math.isnan(x)
    }
    assert faltantes == {3, 4, 11}


def test_extraer_no_redondea_las_coordenadas_a_pixel_entero(video, tmp_path, configuracion):
    resultado = extraer(
        video,
        tmp_path / "salida",
        configuracion,
        crear_detector=lambda _: DetectorFalso(),
        hashear_video=False,
    )
    x = leer_landmarks(resultado.ruta_landmarks).column("x").to_pylist()
    # float32 conserva bastante más que el entero de píxel que usaba apertura_v.py.
    assert x[0] == pytest.approx(0.123456, abs=1e-6)
    assert len({round(valor, 6) for valor in x[:CANTIDAD_LANDMARKS]}) == CANTIDAD_LANDMARKS


def test_metadata_permite_reproducir_la_corrida(video, tmp_path, configuracion):
    resultado = extraer(
        video,
        tmp_path / "salida",
        configuracion,
        crear_detector=lambda _: DetectorFalso(sin_deteccion={7}),
        hashear_video=False,
    )
    metadata = json.loads(resultado.ruta_metadata.read_text(encoding="utf-8"))

    assert metadata["etapa"] == "extraccion"
    assert metadata["video"]["fps"] == 30.0
    assert metadata["video"]["resolucion_inferencia"] == [ANCHO, ALTO]
    assert metadata["modelo"]["sha256_esperado"] == configuracion.modelo.sha256
    assert metadata["codigo"]["version_swimalyzer"]
    assert metadata["entorno"]["mediapipe"]
    assert metadata["configuracion"]["deteccion"] == configuracion.deteccion.model_dump()
    assert metadata["resultado"]["fotogramas_procesados"] == FOTOGRAMAS
    assert metadata["resultado"]["indices_sin_deteccion"] == [7]
    assert metadata["resultado"]["filas_parquet"] == FOTOGRAMAS * CANTIDAD_LANDMARKS


def test_extraer_con_recorte_registra_las_dos_resoluciones(video, tmp_path, configuracion):
    recorte = Recorte(x=0, y=0, ancho=32, alto=24)
    con_recorte = configuracion.model_copy(
        update={"video": configuracion.video.model_copy(update={"recorte": recorte})}
    )
    resultado = extraer(
        video,
        tmp_path / "salida",
        con_recorte,
        crear_detector=lambda _: DetectorFalso(),
        hashear_video=False,
    )
    metadata = json.loads(resultado.ruta_metadata.read_text(encoding="utf-8"))
    assert metadata["video"]["resolucion_original"] == [ANCHO, ALTO]
    assert metadata["video"]["resolucion_inferencia"] == [32, 24]
    assert metadata["video"]["recorte"] == {"x": 0, "y": 0, "ancho": 32, "alto": 24}


def test_extraer_sin_fps_usable_falla_antes_de_dividir_por_cero(video, tmp_path, configuracion):
    with pytest.raises(ErrorDeVideo):
        extraer(
            video,
            tmp_path / "salida",
            configuracion,
            fps_forzado=0.0,
            crear_detector=lambda _: DetectorFalso(),
        )


def test_config_con_modelo_inexistente(tmp_path):
    """El chequeo del modelo no depende de tener el .task descargado."""
    datos = yaml.safe_load(CONFIG_DEL_REPO.read_text(encoding="utf-8"))
    datos["modelo"]["ruta"] = "modelos/no_existe.task"
    archivo = tmp_path / "config.yaml"
    archivo.write_text(yaml.safe_dump(datos), encoding="utf-8")
    assert not cargar_configuracion(archivo).modelo.ruta.is_file()
