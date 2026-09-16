"""Filtrado: interpolación de huecos, Butterworth sin desfase y marcado de intercambios."""

import json

import numpy as np
import pytest

from swimalyzer.config import ErrorDeConfiguracion, cargar_configuracion
from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS, CODO_DER, CODO_IZQ
from swimalyzer.signal.caracterizacion import cargar_series
from swimalyzer.signal.filtrado import (
    ESQUEMA_FILTRADO,
    ErrorDeFiltrado,
    filtrar_corrida,
    filtrar_trayectoria,
    interpolar_huecos_cortos,
    longitud_minima_para_filtfilt,
    marcar_intercambios,
)

from .conftest import ALTO, ANCHO, CONFIG_DEL_REPO, FPS, escribir_corrida

# --- interpolación ---------------------------------------------------------


def test_interpola_los_huecos_cortos_y_deja_los_largos():
    serie = np.array([0.0, 1.0, np.nan, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
    rellena, interpolado = interpolar_huecos_cortos(serie, maximo=3)
    assert rellena[2] == pytest.approx(2.0)
    assert interpolado[2]
    # El hueco de 4 muestras supera el máximo: queda como está.
    assert np.isnan(rellena[5:9]).all()
    assert not interpolado[5:9].any()


def test_no_se_extrapola_en_los_bordes():
    # Sin dos valores alrededor no hay nada que interpolar: prolongar el primero
    # o el último sería inventar movimiento.
    serie = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    rellena, interpolado = interpolar_huecos_cortos(serie, maximo=3)
    assert np.isnan(rellena[0]) and np.isnan(rellena[-1])
    assert not interpolado.any()


def test_maximo_cero_no_interpola_nada():
    serie = np.array([0.0, np.nan, 2.0])
    rellena, interpolado = interpolar_huecos_cortos(serie, maximo=0)
    assert np.isnan(rellena[1])
    assert not interpolado.any()


# --- filtrado --------------------------------------------------------------


def test_longitud_minima_para_filtfilt():
    assert longitud_minima_para_filtfilt(2) == 10
    assert longitud_minima_para_filtfilt(4) == 16


def test_el_filtro_saca_el_ruido_y_conserva_la_señal_lenta():
    tiempo = np.arange(300) / FPS
    limpia = 50 * np.sin(2 * np.pi * 0.5 * tiempo)
    generador = np.random.default_rng(3)
    cruda = limpia + generador.normal(0, 8, tiempo.size)
    suave, filtrado = filtrar_trayectoria(cruda, FPS, orden=2, frecuencia_corte_hz=3.4)
    assert filtrado.all()

    def error(serie):
        return np.sqrt(np.mean((serie - limpia) ** 2))

    # Lo que queda tiene que parecerse bastante más a la señal limpia que la cruda.
    assert error(suave) < error(cruda) / 2


def test_el_filtro_no_corre_la_señal_en_el_tiempo():
    # filtfilt pasa en los dos sentidos: un pico tiene que quedar en su fotograma.
    tiempo = np.arange(300) / FPS
    serie = 50 * np.sin(2 * np.pi * 0.5 * tiempo)
    suave, _ = filtrar_trayectoria(serie, FPS, orden=2, frecuencia_corte_hz=3.4)
    assert int(np.argmax(suave)) == pytest.approx(int(np.argmax(serie)), abs=1)


def test_no_se_filtra_a_traves_de_un_hueco():
    # Dos tramos con valores muy distintos: si el filtro cruzara el hueco, el
    # final del primero se contagiaría del segundo.
    serie = np.concatenate([np.full(40, 10.0), [np.nan] * 5, np.full(40, 400.0)])
    suave, filtrado = filtrar_trayectoria(serie, FPS, orden=2, frecuencia_corte_hz=3.4)
    assert np.isnan(suave[40:45]).all()
    assert not filtrado[40:45].any()
    assert suave[39] == pytest.approx(10.0, abs=1.0)
    assert suave[45] == pytest.approx(400.0, abs=1.0)


def test_un_tramo_demasiado_corto_queda_sin_filtrar_y_marcado():
    serie = np.concatenate([np.arange(5.0), [np.nan] * 3, np.arange(40.0)])
    suave, filtrado = filtrar_trayectoria(serie, FPS, orden=2, frecuencia_corte_hz=3.4)
    assert not filtrado[:5].any()
    # Sin filtrar, pero sin borrar: el dato crudo sigue estando.
    assert np.array_equal(suave[:5], np.arange(5.0))
    assert filtrado[8:].all()


@pytest.mark.parametrize("corte", [0.0, -1.0, 15.0, 40.0])
def test_corte_fuera_del_rango_util_es_error(corte):
    with pytest.raises(ErrorDeFiltrado, match="Nyquist|entre 0"):
        filtrar_trayectoria(np.zeros(50), FPS, orden=2, frecuencia_corte_hz=corte)


# --- corrida completa ------------------------------------------------------


def _corrida_sintetica(tmp_path, fotogramas=120, detectado=None, con_intercambio=False):
    tiempo = np.arange(fotogramas) / FPS
    x = np.tile((0.5 + 0.2 * np.sin(2 * np.pi * 0.5 * tiempo))[:, None], (1, CANTIDAD_LANDMARKS))
    y = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.5)
    visibility = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.8)
    # Un landmark con visibility muy baja: no tiene que desaparecer del resultado.
    visibility[:, CODO_DER] = 0.05
    x[:, CODO_IZQ] = 100 / ANCHO
    x[:, CODO_DER] = 300 / ANCHO
    if con_intercambio:
        x[60:, CODO_IZQ], x[60:, CODO_DER] = x[60:, CODO_DER].copy(), x[60:, CODO_IZQ].copy()
    return escribir_corrida(tmp_path / "corrida", x, y, visibility, detectado)


@pytest.fixture
def configuracion():
    return cargar_configuracion(CONFIG_DEL_REPO)


def test_filtrar_corrida_escribe_el_parquet_con_su_esquema(tmp_path, configuracion):
    import pyarrow.parquet as pq

    corrida = _corrida_sintetica(tmp_path)
    resultado = filtrar_corrida(corrida, tmp_path / "salida", configuracion)
    tabla = pq.read_table(resultado.ruta_filtrado)
    assert tabla.schema.names == ESQUEMA_FILTRADO.names
    assert tabla.num_rows == 120 * CANTIDAD_LANDMARKS


def test_el_filtrado_no_descarta_por_visibility(tmp_path, configuracion):
    import pyarrow.parquet as pq

    corrida = _corrida_sintetica(tmp_path)
    resultado = filtrar_corrida(corrida, tmp_path / "salida", configuracion)
    datos = pq.read_table(resultado.ruta_filtrado).to_pandas()
    codo_der = datos[datos.landmark_id == CODO_DER]
    # visibility 0.05, muy por debajo de cualquier umbral: sigue entero y filtrado.
    assert not codo_der.x_px.isna().any()
    assert codo_der.filtrado.all()
    # Y la visibility viaja con el dato, para que la etapa de métricas la use.
    assert codo_der.visibility.max() == pytest.approx(0.05, abs=1e-6)


def test_las_coordenadas_quedan_en_pixeles(tmp_path, configuracion):
    import pyarrow.parquet as pq

    corrida = _corrida_sintetica(tmp_path)
    resultado = filtrar_corrida(corrida, tmp_path / "salida", configuracion)
    datos = pq.read_table(resultado.ruta_filtrado).to_pandas()
    codo = datos[datos.landmark_id == CODO_IZQ]
    assert codo.x_px.median() == pytest.approx(100.0, abs=1.0)
    assert codo.y_px.median() == pytest.approx(0.5 * ALTO, abs=1.0)


def test_los_huecos_cortos_se_interpolan_y_los_largos_no(tmp_path, configuracion):
    detectado = np.ones(120, bool)
    detectado[30:32] = False  # hueco corto: 2 fotogramas
    detectado[60:80] = False  # hueco largo: 20 fotogramas
    corrida = _corrida_sintetica(tmp_path, detectado=detectado)
    resultado = filtrar_corrida(corrida, tmp_path / "salida", configuracion)
    assert resultado.fotogramas_interpolados == 2
    assert resultado.muestras_interpoladas == 2 * CANTIDAD_LANDMARKS
    assert resultado.muestras_con_dato == (120 - 20) * CANTIDAD_LANDMARKS


def test_marcar_intercambios_marca_los_dos_landmarks_del_par(tmp_path):
    corrida = _corrida_sintetica(tmp_path, con_intercambio=True)
    series = cargar_series(corrida)
    banderas, tasas = marcar_intercambios(series, margen_minimo_px=20.0, separacion_minima_px=20.0)
    assert banderas[60, CODO_IZQ] and banderas[60, CODO_DER]
    assert banderas.sum() == 2
    par = "codo_izq/codo_der"
    assert tasas[par]["fotogramas_marcados"] == 1
    assert tasas[par]["fotogramas"] == [60]
    assert tasas["muneca_izq/muneca_der"]["fotogramas_marcados"] == 0


def test_la_metadata_del_filtrado_encadena_con_la_extraccion(tmp_path, configuracion):
    corrida = _corrida_sintetica(tmp_path)
    resultado = filtrar_corrida(corrida, tmp_path / "salida", configuracion)
    metadata = json.loads(resultado.ruta_metadata.read_text(encoding="utf-8"))
    assert metadata["etapa"] == "filtrado"
    assert metadata["origen"]["metadata_extraccion"]["etapa"] == "extraccion"
    assert len(metadata["origen"]["sha256_landmarks"]) == 64
    parametros = metadata["parametros"]
    assert parametros["gate_por_visibility"] is False
    assert parametros["frecuencia_corte_hz"] == configuracion.filtrado.frecuencia_corte_hz
    assert parametros["orden"] == configuracion.filtrado.orden
    assert parametros["metodo_correccion_intercambios"] == "ninguno"
    assert metadata["resultado"]["intercambios_por_par"]


def test_filtrado_con_tipo_no_soportado(tmp_path, configuracion):
    corrida = _corrida_sintetica(tmp_path)
    otra = configuracion.model_copy(
        update={"filtrado": configuracion.filtrado.model_copy(update={"tipo": "kalman"})}
    )
    with pytest.raises(ErrorDeConfiguracion, match="no está implementado"):
        filtrar_corrida(corrida, tmp_path / "salida", otra)


def test_filtrado_con_decision_en_null_se_niega_a_correr(tmp_path, configuracion):
    corrida = _corrida_sintetica(tmp_path)
    otra = configuracion.model_copy(
        update={"filtrado": configuracion.filtrado.model_copy(update={"frecuencia_corte_hz": None})}
    )
    with pytest.raises(ErrorDeConfiguracion, match="está en null"):
        filtrar_corrida(corrida, tmp_path / "salida", otra)
