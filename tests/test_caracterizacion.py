"""Caracterización de la señal: conversión a píxeles, huecos, espectro e intercambios."""

import numpy as np
import pytest

from swimalyzer.pose.landmarks import CANTIDAD_LANDMARKS, CODO_DER, CODO_IZQ
from swimalyzer.signal.caracterizacion import (
    analisis_de_residuos,
    candidatos_a_intercambio,
    cargar_series,
    descarte_por_umbral,
    espectro,
    frecuencia_de_potencia_acumulada,
    huecos,
    percentiles_de_visibility,
    tramos_continuos,
)

from .conftest import ALTO, ANCHO, FPS, escribir_corrida


@pytest.fixture
def corrida(tmp_path):
    fotogramas = 120
    generador = np.random.default_rng(0)
    x = generador.uniform(0.2, 0.8, (fotogramas, CANTIDAD_LANDMARKS))
    y = generador.uniform(0.2, 0.8, (fotogramas, CANTIDAD_LANDMARKS))
    visibility = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.6)
    visibility[:, CODO_DER] = 0.2
    detectado = np.ones(fotogramas, bool)
    detectado[10:15] = False
    return escribir_corrida(tmp_path / "corrida", x, y, visibility, detectado), x, y, detectado


def test_cargar_series_convierte_a_pixeles_con_la_resolucion_de_la_metadata(corrida):
    directorio, x, y, detectado = corrida
    series = cargar_series(directorio)
    # La conversión a píxeles usa ancho y alto por separado: si se hiciera sobre
    # coordenadas normalizadas, cualquier ángulo saldría deformado.
    assert series.x[0, CODO_IZQ] == pytest.approx(x[0, CODO_IZQ] * ANCHO, rel=1e-5)
    assert series.y[0, CODO_IZQ] == pytest.approx(y[0, CODO_IZQ] * ALTO, rel=1e-5)
    assert series.ancho == ANCHO and series.alto == ALTO
    assert series.fps == FPS
    assert np.array_equal(series.detectado, detectado)
    assert series.cantidad_fotogramas == x.shape[0]


def test_los_fotogramas_sin_deteccion_quedan_como_hueco(corrida):
    series = cargar_series(corrida[0])
    assert huecos(series.detectado) == [(10, 15)]
    assert tramos_continuos(series.detectado) == [(0, 10), (15, 120)]


def test_percentiles_de_visibility_ignoran_los_fotogramas_sin_deteccion(corrida):
    series = cargar_series(corrida[0])
    resumen = percentiles_de_visibility(series, CODO_IZQ)
    assert resumen["n"] == int(series.detectado.sum())
    assert resumen["p50"] == pytest.approx(0.6, abs=1e-6)


def test_descarte_por_umbral_informa_los_dos_denominadores(corrida):
    series = cargar_series(corrida[0])
    fotogramas = series.cantidad_fotogramas
    detectados = int(series.detectado.sum())

    pasa = descarte_por_umbral(series, CODO_IZQ, 0.5)
    # Visibility 0.6 pasa el umbral: lo único descartado son los huecos.
    assert pasa["descartados_sobre_detectados"] == pytest.approx(0.0)
    assert pasa["descartados_sobre_todos"] == pytest.approx(1 - detectados / fotogramas)

    no_pasa = descarte_por_umbral(series, CODO_DER, 0.5)
    assert no_pasa["descartados_sobre_todos"] == pytest.approx(1.0)
    assert no_pasa["fotogramas_usables"] == 0


# --- tramos ----------------------------------------------------------------


def test_tramos_continuos_respeta_el_largo_minimo():
    mascara = np.array([1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert tramos_continuos(mascara) == [(0, 2), (3, 6), (7, 8)]
    assert tramos_continuos(mascara, largo_minimo=3) == [(3, 6)]
    assert tramos_continuos(np.zeros(5, bool)) == []
    assert tramos_continuos(np.ones(5, bool)) == [(0, 5)]


# --- espectro y residuos ---------------------------------------------------


def test_el_espectro_encuentra_la_frecuencia_de_una_senal_conocida():
    tiempo = np.arange(256) / FPS
    serie = 50 + 20 * np.sin(2 * np.pi * 2.0 * tiempo)
    frecuencias, potencia = espectro(serie, FPS, nperseg=256)
    assert frecuencias[int(np.argmax(potencia))] == pytest.approx(2.0, abs=0.15)
    # La media no aporta potencia: interesa el movimiento, no dónde está el punto.
    assert potencia[0] < potencia.max() / 100


def test_potencia_acumulada_crece_con_la_fraccion_pedida():
    tiempo = np.arange(256) / FPS
    serie = np.sin(2 * np.pi * 1.0 * tiempo) + 0.1 * np.sin(2 * np.pi * 8.0 * tiempo)
    frecuencias, potencia = espectro(serie, FPS, nperseg=256)
    assert frecuencia_de_potencia_acumulada(frecuencias, potencia, 0.5) <= 1.5
    assert frecuencia_de_potencia_acumulada(
        frecuencias, potencia, 0.99
    ) >= frecuencia_de_potencia_acumulada(frecuencias, potencia, 0.5)


@pytest.mark.parametrize("funcion", [espectro, analisis_de_residuos])
def test_espectro_y_residuos_rechazan_tramos_con_huecos(funcion):
    serie = np.array([1.0, 2.0, np.nan, 4.0] * 16)
    with pytest.raises(ValueError, match="hueco"):
        funcion(serie, FPS)


def test_el_analisis_de_residuos_separa_senal_de_ruido():
    # Señal lenta de 1 Hz más ruido blanco: el corte tiene que quedar por
    # encima de la señal y bastante por debajo de Nyquist.
    generador = np.random.default_rng(1)
    tiempo = np.arange(600) / FPS
    limpia = 30 * np.sin(2 * np.pi * 1.0 * tiempo)
    ruido_rms = 2.0
    residuos = analisis_de_residuos(limpia + generador.normal(0, ruido_rms, tiempo.size), FPS)
    assert 1.0 < residuos.corte_optimo_hz < 8.0
    assert residuos.ruido_estimado_px == pytest.approx(ruido_rms, rel=0.5)
    # El residuo cae al subir el corte: la recta de la zona de ruido baja.
    assert residuos.pendiente < 0


# --- intercambios ----------------------------------------------------------


def _series_de_dos_puntos(tmp_path, x_izq, x_der):
    fotogramas = x_izq.size
    x = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.5)
    y = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.5)
    visibility = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.9)
    x[:, CODO_IZQ] = x_izq / ANCHO
    x[:, CODO_DER] = x_der / ANCHO
    y[:, CODO_IZQ] = 0.4
    y[:, CODO_DER] = 0.4
    return cargar_series(escribir_corrida(tmp_path / "dos_puntos", x, y, visibility))


def test_no_marca_nada_cuando_cada_punto_sigue_el_suyo(tmp_path):
    fotogramas = 60
    izq = 100 + 2.0 * np.arange(fotogramas)
    der = 300 - 2.0 * np.arange(fotogramas) * 0.1
    series = _series_de_dos_puntos(tmp_path, izq, der)
    candidatos = candidatos_a_intercambio(series, CODO_IZQ, CODO_DER)
    assert candidatos.fotogramas.size == 0
    assert candidatos.transiciones_evaluadas == fotogramas - 1


def test_marca_el_fotograma_en_el_que_se_intercambian_las_etiquetas(tmp_path):
    fotogramas = 60
    izq = np.full(fotogramas, 100.0)
    der = np.full(fotogramas, 300.0)
    # A partir del fotograma 30 el modelo cambia las etiquetas de lado.
    izq[30:], der[30:] = der[30:].copy(), izq[30:].copy()
    series = _series_de_dos_puntos(tmp_path, izq, der)
    candidatos = candidatos_a_intercambio(series, CODO_IZQ, CODO_DER)
    assert candidatos.fotogramas.tolist() == [30]
    assert candidatos.margen_px[0] == pytest.approx(400.0)
    assert candidatos.cruces_en_x == 1


def test_no_marca_intercambios_entre_puntos_encimados(tmp_path):
    # Con los dos puntos casi en el mismo lugar, cuál es cuál no cambia
    # ninguna medición: marcar ahí sería ruido.
    fotogramas = 60
    izq = np.full(fotogramas, 200.0)
    der = np.full(fotogramas, 203.0)
    izq[30:], der[30:] = der[30:].copy(), izq[30:].copy()
    series = _series_de_dos_puntos(tmp_path, izq, der)
    assert candidatos_a_intercambio(series, CODO_IZQ, CODO_DER).fotogramas.size == 0
