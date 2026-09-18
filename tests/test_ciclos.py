"""Segmentación en ciclos: dónde corta, qué no cruza, y qué no descarta."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from swimalyzer.config import ErrorDeConfiguracion, cargar_configuracion
from swimalyzer.metrics.angulos import (
    ARTICULACIONES,
    MOTIVOS,
    SerieDeAngulo,
    calcular_angulos_de_corrida,
)
from swimalyzer.metrics.ciclos import (
    ESQUEMA_CICLOS,
    MALLA_PORCENTAJE,
    SENALES_DE_SEGMENTACION,
    Ciclo,
    construir_senal,
    curva_media,
    normalizar,
    normalizar_bandera,
    segmentar,
    segmentar_corrida,
)
from swimalyzer.pose.landmarks import (
    CADERA_IZQ,
    CANTIDAD_LANDMARKS,
    CODO_IZQ,
    HOMBRO_IZQ,
    MUNECA_IZQ,
    RODILLA_IZQ,
    TOBILLO_IZQ,
)
from swimalyzer.signal.filtrado import cargar_filtrado, filtrar_corrida

from .conftest import ALTO, ANCHO, CONFIG_DEL_REPO, FPS, escribir_corrida

# La brazada sintética: 2 s de período, con el primer máximo de altura en el
# fotograma 15 para que ninguno caiga sobre el borde de la serie.
PERIODO_S = 2.0
PRIMER_MAXIMO = 15
FOTOGRAMAS = 300


@pytest.fixture
def configuracion():
    return cargar_configuracion(CONFIG_DEL_REPO)


def _corrida_filtrada(tmp_path, configuracion, visibility_muneca=0.9, detectado=None):
    """Una corrida donde la muñeca sube y baja alrededor del hombro, a 0.5 Hz.

    El hombro está fijo en y=100 px y la muñeca oscila entre y=60 (40 px por
    encima del hombro) e y=140. La altura de la muñeca sobre el hombro es
    entonces una sinusoide de máximos conocidos, así que los cortes son
    verificables a mano.
    """
    tiempo = np.arange(FOTOGRAMAS) / FPS
    x = np.full((FOTOGRAMAS, CANTIDAD_LANDMARKS), 0.5)
    y = np.full((FOTOGRAMAS, CANTIDAD_LANDMARKS), 0.5)
    visibility = np.full((FOTOGRAMAS, CANTIDAD_LANDMARKS), 0.9)

    def poner(landmark, px, py):
        x[:, landmark] = px / ANCHO
        y[:, landmark] = py / ALTO

    poner(HOMBRO_IZQ, 100.0, 100.0)
    poner(CODO_IZQ, 200.0, 100.0)
    poner(CADERA_IZQ, 100.0, 200.0)
    poner(RODILLA_IZQ, 100.0, 250.0)
    poner(TOBILLO_IZQ, 100.0, 300.0)

    fase = 2 * np.pi * (tiempo - PRIMER_MAXIMO / FPS) / PERIODO_S
    x[:, MUNECA_IZQ] = 250.0 / ANCHO
    y[:, MUNECA_IZQ] = (100.0 - 40.0 * np.cos(fase)) / ALTO
    visibility[:, MUNECA_IZQ] = visibility_muneca

    corrida = escribir_corrida(tmp_path / "corrida", x, y, visibility, detectado)
    filtrar_corrida(corrida, tmp_path / "filtrada", configuracion)
    calcular_angulos_de_corrida(tmp_path / "filtrada", tmp_path / "filtrada", configuracion)
    return tmp_path / "filtrada"


# --- el signo de la señal --------------------------------------------------


def test_la_senal_es_positiva_con_la_muneca_por_encima_del_hombro(tmp_path, configuracion):
    """Pin del signo: en coordenadas de imagen la `y` crece hacia abajo.

    Si la resta se diera vuelta, el máximo sería la mano más profunda del tirón
    en vez de la más alta del recobro, y toda la segmentación cambiaría de
    evento sin que nada fallara.
    """
    corrida = _corrida_filtrada(tmp_path, configuracion)
    series = cargar_filtrado(corrida)
    senal = construir_senal(series, "muneca_y_rel_hombro")

    arriba = series.y[:, MUNECA_IZQ] < series.y[:, HOMBRO_IZQ]
    assert (senal[arriba] > 0).all()
    assert (senal[~arriba] <= 0).all()


def test_una_senal_desconocida_falla_nombrando_las_que_hay(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    series = cargar_filtrado(corrida)
    with pytest.raises(ErrorDeConfiguracion, match="muneca_y_rel_hombro"):
        construir_senal(series, "codo_y_rel_hombro")


# --- dónde caen los cortes -------------------------------------------------


def test_los_cortes_caen_en_los_maximos_conocidos(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = segmentar_corrida(corrida, tmp_path / "ciclos", configuracion)

    esperados = list(range(PRIMER_MAXIMO, FOTOGRAMAS, int(PERIODO_S * FPS)))
    cortes = sorted(
        {ciclo.inicio for ciclo in resultado.ciclos} | {ciclo.fin for ciclo in resultado.ciclos}
    )
    assert cortes == esperados
    assert len(resultado.ciclos) == len(esperados) - 1
    for ciclo in resultado.ciclos:
        assert ciclo.duracion_s(FPS) == pytest.approx(PERIODO_S)
    assert resultado.frecuencia_de_brazada_hz == pytest.approx(1 / PERIODO_S)


def test_una_distancia_minima_mayor_al_periodo_se_saltea_maximos(tmp_path, configuracion):
    """La distancia mínima es lo único que restringe: subirla tiene que costar ciclos."""
    corrida = _corrida_filtrada(tmp_path, configuracion)
    holgada = configuracion.model_copy(
        update={
            "segmentacion": configuracion.segmentacion.model_copy(
                update={"distancia_minima_entre_picos_s": PERIODO_S * 1.5}
            )
        }
    )
    resultado = segmentar_corrida(corrida, tmp_path / "ciclos", holgada)
    assert all(ciclo.duracion_s(FPS) >= PERIODO_S * 1.5 for ciclo in resultado.ciclos)


def test_ningun_ciclo_cruza_un_hueco(tmp_path, configuracion):
    # Un hueco de 40 fotogramas en el medio, más largo que el máximo
    # interpolable, así que corta el tramo de verdad.
    detectado = np.ones(FOTOGRAMAS, bool)
    detectado[130:170] = False
    corrida = _corrida_filtrada(tmp_path, configuracion, detectado=detectado)
    resultado = segmentar_corrida(corrida, tmp_path / "ciclos", configuracion)

    assert resultado.ciclos, "el hueco no debería dejar la serie sin ciclos"
    for ciclo in resultado.ciclos:
        assert not (ciclo.inicio < 170 and ciclo.fin > 130), (
            f"el ciclo {ciclo.numero} ({ciclo.inicio}-{ciclo.fin}) cruza el hueco"
        )


def test_los_cortes_del_borde_no_arman_ciclo():
    """Entre el comienzo de un tramo y su primer corte hay un trozo, no un ciclo."""
    senal = np.zeros(100)
    senal[[20, 60]] = 1.0
    disponible = np.ones(100, bool)
    ciclos = segmentar(senal, disponible, FPS, 0.5)
    assert [(ciclo.inicio, ciclo.fin) for ciclo in ciclos] == [(20, 60)]


# --- normalización ---------------------------------------------------------


def test_normalizar_lleva_los_extremos_del_ciclo_al_0_y_al_100():
    valores = np.arange(0.0, 61.0)
    ciclo = Ciclo(numero=0, tramo=0, inicio=10, fin=40)
    curva = normalizar(valores, ciclo)
    assert curva.size == MALLA_PORCENTAJE.size
    assert curva[0] == pytest.approx(10.0)
    assert curva[-1] == pytest.approx(40.0)
    # La rampa es lineal, así que el 50 % tiene que caer en el punto medio.
    assert curva[50] == pytest.approx(25.0)


def test_la_bandera_se_propaga_por_vecino_y_no_se_promedia():
    banderas = np.zeros(21, bool)
    banderas[:11] = True
    ciclo = Ciclo(numero=0, tramo=0, inicio=0, fin=20)
    llevada = normalizar_bandera(banderas, ciclo)
    assert llevada.dtype == bool
    assert llevada[0] and llevada[50]
    assert not llevada[-1]


def test_dos_ciclos_de_distinta_duracion_se_comparan_en_la_misma_malla():
    """Es lo que hace que promediar ciclos tenga sentido: compara fases, no fotogramas."""
    corto = Ciclo(numero=0, tramo=0, inicio=0, fin=20)
    largo = Ciclo(numero=1, tramo=0, inicio=0, fin=80)
    # La misma rampa recorrida en 21 y en 81 fotogramas tiene que dar la misma
    # curva normalizada.
    serie_corta = np.zeros(100)
    serie_corta[0:21] = np.linspace(0, 1, 21)
    serie_larga = np.zeros(100)
    serie_larga[0:81] = np.linspace(0, 1, 81)
    assert normalizar(serie_corta, corto) == pytest.approx(normalizar(serie_larga, largo), abs=1e-9)


# --- qué no se descarta ----------------------------------------------------


def test_los_ciclos_con_mediciones_marcadas_entran_y_la_cobertura_lo_dice(tmp_path, configuracion):
    # Visibility de la muñeca por debajo del umbral de reporte: todo el ángulo
    # de codo queda marcado.
    corrida = _corrida_filtrada(tmp_path, configuracion, visibility_muneca=0.1)
    resultado = segmentar_corrida(corrida, tmp_path / "ciclos", configuracion)

    codo = resultado.curvas["codo_izq"]
    assert codo.n == len(resultado.ciclos), "no se descartó ningún ciclo por estar marcado"
    assert codo.cobertura_marcada == pytest.approx(1.0)
    # Y la curva se calculó igual: marcar no es descartar.
    assert not np.isnan(codo.media).any()


def _serie_sintetica(grados, marcado=None):
    """Una SerieDeAngulo de codo armada a mano, para probar la curva sin pipeline."""
    grados = np.asarray(grados, dtype=float)
    marcado = np.zeros(grados.size, bool) if marcado is None else marcado
    articulacion = next(art for art in ARTICULACIONES if art.nombre == "codo_izq")
    return SerieDeAngulo(
        articulacion=articulacion,
        grados=grados,
        visibility_minima=np.full(grados.size, 0.9),
        motivos={motivo: np.zeros(grados.size, bool) for motivo in MOTIVOS}
        | {"visibility_baja": marcado},
    )


def test_un_ciclo_con_un_hueco_de_angulo_adentro_no_entra_en_la_curva():
    """No se puede normalizar un ciclo sin inventar el trozo que falta."""
    grados = np.linspace(90.0, 180.0, 100)
    grados[45] = np.nan
    serie = _serie_sintetica(grados)
    con_hueco = Ciclo(numero=0, tramo=0, inicio=30, fin=60)
    limpio = Ciclo(numero=1, tramo=0, inicio=60, fin=90)

    curva = curva_media(serie, [con_hueco, limpio])
    assert [ciclo.numero for ciclo in curva.ciclos] == [1]
    assert not np.isnan(curva.curvas).any()


def test_el_desvio_de_un_solo_ciclo_es_nan_y_no_cero():
    """Con un ciclo no hay dispersión que estimar: decir 0 sería afirmar de más."""
    serie = _serie_sintetica(np.linspace(90.0, 180.0, 100))
    curva = curva_media(serie, [Ciclo(numero=0, tramo=0, inicio=20, fin=80)])
    assert curva.n == 1
    assert np.isnan(curva.desvio).all()


# --- persistencia ----------------------------------------------------------


def test_el_parquet_de_ciclos_tiene_su_esquema_y_una_fila_por_punto(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = segmentar_corrida(corrida, tmp_path / "ciclos", configuracion)
    tabla = pq.read_table(resultado.ruta_ciclos)

    assert tabla.schema.names == ESQUEMA_CICLOS.names
    esperadas = sum(curva.n for curva in resultado.curvas.values()) * MALLA_PORCENTAJE.size
    assert tabla.num_rows == esperadas


def test_la_metadata_registra_la_senal_el_evento_y_los_tramos(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = segmentar_corrida(corrida, tmp_path / "ciclos", configuracion)
    metadata = json.loads(resultado.ruta_metadata.read_text(encoding="utf-8"))

    parametros = metadata["parametros"]
    assert parametros["senal"] == "muneca_y_rel_hombro"
    assert parametros["evento_de_corte"] == SENALES_DE_SEGMENTACION["muneca_y_rel_hombro"].evento
    # Sin esta lista, el número de tramo de un ciclo no ubica nada.
    assert metadata["resultado"]["tramos_continuos"]
    assert metadata["resultado"]["frecuencia_de_brazada_hz"] == pytest.approx(1 / PERIODO_S)


def test_la_segmentacion_se_niega_a_correr_con_la_decision_en_null(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    sin_decidir = configuracion.model_copy(
        update={"segmentacion": configuracion.segmentacion.model_copy(update={"senal": None})}
    )
    with pytest.raises(ErrorDeConfiguracion, match="segmentacion.senal"):
        segmentar_corrida(corrida, tmp_path / "ciclos", sin_decidir)
