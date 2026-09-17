"""Ángulos articulares: geometría, herencia de banderas y persistencia."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from swimalyzer.config import ErrorDeConfiguracion, cargar_configuracion
from swimalyzer.metrics.angulos import (
    ARTICULACIONES,
    ESQUEMA_ANGULOS,
    MOTIVOS,
    angulo_interior,
    calcular_angulos_de_corrida,
    cargar_series_de_angulo,
    rango,
    resumir,
    salto_adyacente,
    velocidad_angular,
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
from swimalyzer.signal.filtrado import filtrar_corrida

from .conftest import ALTO, ANCHO, CONFIG_DEL_REPO, FPS, escribir_corrida

# --- geometría -------------------------------------------------------------


def _punto(x, y):
    return np.array([[x, y]], dtype=float)


@pytest.mark.parametrize(
    ("proximal", "distal", "esperado"),
    [
        ((1.0, 0.0), (0.0, 1.0), 90.0),
        ((1.0, 0.0), (-1.0, 0.0), 180.0),
        ((1.0, 0.0), (1.0, 0.0), 0.0),
        ((1.0, 0.0), (1.0, 1.0), 45.0),
        # El ángulo no depende de la distancia a la que estén los extremos.
        ((100.0, 0.0), (0.0, 3.0), 90.0),
    ],
)
def test_angulo_interior_en_geometrias_conocidas(proximal, distal, esperado):
    medido = angulo_interior(_punto(*proximal), _punto(0.0, 0.0), _punto(*distal))
    assert medido[0] == pytest.approx(esperado)


def test_el_angulo_nunca_pasa_de_180_ni_baja_de_cero():
    generador = np.random.default_rng(7)
    puntos = generador.uniform(-500, 500, (400, 3, 2))
    medido = angulo_interior(puntos[:, 0], puntos[:, 1], puntos[:, 2])
    assert np.all((medido >= 0) & (medido <= 180))


def test_un_segmento_de_largo_cero_no_define_angulo():
    # El vértice encima de un extremo: no hay dirección, así que no hay ángulo.
    medido = angulo_interior(_punto(10.0, 10.0), _punto(10.0, 10.0), _punto(0.0, 0.0))
    assert np.isnan(medido[0])


def test_calcular_sobre_coordenadas_normalizadas_distorsiona_el_angulo():
    # La razón por la que todo esto se hace en píxeles: con 576x324 la relación
    # de aspecto no es 1 y el mismo ángulo real da distinto si no se convierte.
    # Un ángulo de 45°: un lado horizontal y el otro en diagonal. La diagonal
    # es la que se deforma al normalizar, porque x y y se dividen por números
    # distintos.
    vertice_px, proximal_px, distal_px = (100.0, 100.0), (200.0, 100.0), (200.0, 200.0)
    en_pixeles = angulo_interior(_punto(*proximal_px), _punto(*vertice_px), _punto(*distal_px))[0]

    def normalizar(punto):
        return _punto(punto[0] / ANCHO, punto[1] / ALTO)

    normalizado = angulo_interior(
        normalizar(proximal_px), normalizar(vertice_px), normalizar(distal_px)
    )[0]
    assert en_pixeles == pytest.approx(45.0)
    assert abs(normalizado - en_pixeles) > 10


# --- velocidad angular -----------------------------------------------------


def test_velocidad_angular_es_el_cambio_respecto_del_fotograma_anterior():
    grados = np.array([100.0, 110.0, 105.0, 105.0])
    medida = velocidad_angular(grados)
    # El primer fotograma no tiene con qué compararse.
    assert np.isnan(medida[0])
    np.testing.assert_allclose(medida[1:], [10.0, 5.0, 0.0])


def test_la_velocidad_no_cruza_un_hueco():
    # Un salto a través de un hueco no es velocidad: es todo lo que pasó
    # mientras no hubo medición.
    grados = np.array([100.0, np.nan, 20.0, 25.0])
    medida = velocidad_angular(grados)
    assert np.isnan(medida[1]) and np.isnan(medida[2])
    assert medida[3] == pytest.approx(5.0)


def test_el_salto_adyacente_mira_el_que_entra_y_el_que_sale():
    # Un valor aislado y disparatado: entra con 80 y sale con 80.
    grados = np.array([100.0, 100.0, 20.0, 100.0, 100.0])
    salto = salto_adyacente(velocidad_angular(grados))
    assert salto[2] == pytest.approx(80.0)
    # El fotograma anterior al artefacto también lo ve, por el salto que sale.
    assert salto[1] == pytest.approx(80.0)
    assert salto[0] == pytest.approx(0.0)


# --- corrida completa ------------------------------------------------------


@pytest.fixture
def configuracion():
    return cargar_configuracion(CONFIG_DEL_REPO)


def _corrida_filtrada(tmp_path, configuracion, visibility_codo=0.8, detectado=None):
    """Arma y filtra una corrida donde el brazo izquierdo flexiona y extiende.

    El hombro y la cadera quedan quietos; el codo describe un ángulo que va de
    90° a 180°, así que el resultado es verificable a mano.
    """
    fotogramas = 120
    tiempo = np.arange(fotogramas) / FPS
    x = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.5)
    y = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.5)
    visibility = np.full((fotogramas, CANTIDAD_LANDMARKS), 0.9)

    # Hombro, codo, cadera, rodilla y tobillo fijos, en píxeles convertidos a
    # normalizadas al escribir la corrida.
    def poner(landmark, px, py, cuadro=slice(None)):
        x[cuadro, landmark] = px / ANCHO
        y[cuadro, landmark] = py / ALTO

    poner(HOMBRO_IZQ, 100.0, 100.0)
    poner(CODO_IZQ, 200.0, 100.0)
    poner(CADERA_IZQ, 100.0, 200.0)
    poner(RODILLA_IZQ, 100.0, 250.0)
    poner(TOBILLO_IZQ, 100.0, 300.0)

    # La muñeca gira alrededor del codo: el ángulo de codo va de 90° a 180°.
    apertura = np.radians(90 + 90 * (1 + np.sin(2 * np.pi * 0.5 * tiempo)) / 2)
    x[:, MUNECA_IZQ] = (200.0 + 100.0 * np.cos(np.pi - apertura)) / ANCHO
    y[:, MUNECA_IZQ] = (100.0 + 100.0 * np.sin(np.pi - apertura)) / ALTO
    visibility[:, MUNECA_IZQ] = visibility_codo

    corrida = escribir_corrida(tmp_path / "corrida", x, y, visibility, detectado)
    filtrar_corrida(corrida, tmp_path / "filtrada", configuracion)
    return tmp_path / "filtrada"


def _con_rango_de_codo(configuracion, minimo: float):
    rangos = dict(configuracion.calidad.rango_anatomico_grados)
    rangos["codo"] = rangos["codo"].model_copy(update={"minimo": minimo})
    return configuracion.model_copy(
        update={
            "calidad": configuracion.calidad.model_copy(update={"rango_anatomico_grados": rangos})
        }
    )


def _con_velocidad_maxima(configuracion, grados_por_fotograma: float):
    return configuracion.model_copy(
        update={
            "calidad": configuracion.calidad.model_copy(
                update={"velocidad_angular_maxima_grados_por_fotograma": grados_por_fotograma}
            )
        }
    )


def test_el_parquet_de_angulos_tiene_su_esquema_y_una_fila_por_articulacion(
    tmp_path, configuracion
):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)
    tabla = pq.read_table(resultado.ruta_angulos)
    assert tabla.schema.names == ESQUEMA_ANGULOS.names
    assert tabla.num_rows == 120 * len(ARTICULACIONES)


def test_el_angulo_medido_es_el_que_dicta_la_geometria(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)
    codo = resultado.series["codo_izq"].grados
    # La muñeca barre de 90° a 180° alrededor del codo; el filtro suaviza los
    # extremos, así que se comparan con tolerancia.
    assert np.nanmin(codo) == pytest.approx(90.0, abs=3.0)
    assert np.nanmax(codo) == pytest.approx(180.0, abs=3.0)
    # Hombro y cadera están en ángulo recto respecto del codo, y no se mueven.
    assert resultado.series["hombro_izq"].grados == pytest.approx(90.0, abs=1.0)
    # Cadera, rodilla y tobillo están alineados: la pierna está extendida.
    assert resultado.series["rodilla_izq"].grados == pytest.approx(180.0, abs=1.0)


def test_el_angulo_hereda_la_visibility_baja_de_su_peor_landmark(tmp_path, configuracion):
    # Solo la muñeca tiene visibility por debajo del umbral de reporte (0.3).
    corrida = _corrida_filtrada(tmp_path, configuracion, visibility_codo=0.1)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)

    codo = resultado.series["codo_izq"]
    assert codo.motivos["visibility_baja"].all()
    assert codo.marcado.all()
    # El valor se calcula igual: marcar no es descartar.
    assert not np.isnan(codo.grados).any()
    assert codo.visibility_minima == pytest.approx(0.1, abs=1e-5)

    # El hombro no usa la muñeca, así que no se contagia.
    hombro = resultado.series["hombro_izq"]
    assert not hombro.motivos["visibility_baja"].any()
    assert not hombro.marcado.any()


def test_sin_los_tres_landmarks_no_hay_angulo_y_el_hueco_queda_registrado(tmp_path, configuracion):
    detectado = np.ones(120, bool)
    detectado[40:60] = False  # hueco largo: no se interpola
    corrida = _corrida_filtrada(tmp_path, configuracion, detectado=detectado)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)

    codo = resultado.series["codo_izq"]
    assert np.isnan(codo.grados[40:60]).all()
    assert not codo.con_dato[40:60].any()
    # Un fotograma sin ángulo no se marca: no hay nada que calificar.
    assert not codo.marcado[40:60].any()
    resumen = resultado.resumenes["codo_izq"]
    assert resumen.fotogramas == 120
    assert resumen.con_dato == 100


def test_el_angulo_hereda_la_interpolacion(tmp_path, configuracion):
    detectado = np.ones(120, bool)
    detectado[50:52] = False  # hueco corto: se interpola
    corrida = _corrida_filtrada(tmp_path, configuracion, detectado=detectado)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)

    codo = resultado.series["codo_izq"]
    assert codo.con_dato[50:52].all()
    assert codo.motivos["interpolado"][50:52].all()
    assert not codo.motivos["interpolado"][:50].any()


def test_un_angulo_fuera_del_rango_anatomico_queda_marcado(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    # El codo barre de 90° a 180°: con el piso en 120° tiene que marcar la parte
    # cerrada del recorrido y dejar intacta la abierta.
    otra = _con_rango_de_codo(configuracion, minimo=120.0)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", otra)

    codo = resultado.series["codo_izq"]
    fuera = codo.motivos["fuera_de_rango_en_el_plano_medido"]
    assert fuera.any() and not fuera.all()
    np.testing.assert_array_equal(fuera, codo.con_dato & (codo.grados < 120.0))
    # El valor sigue estando: marcar no es descartar.
    assert not np.isnan(codo.grados[fuera]).any()
    # Y la rodilla, que está extendida en 180°, no se contagia.
    assert not resultado.series["rodilla_izq"].motivos["fuera_de_rango_en_el_plano_medido"].any()


def test_el_rango_se_busca_por_tipo_de_articulacion_y_su_falta_es_un_error(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    rangos = dict(configuracion.calidad.rango_anatomico_grados)
    del rangos["codo"]
    otra = configuracion.model_copy(
        update={
            "calidad": configuracion.calidad.model_copy(update={"rango_anatomico_grados": rangos})
        }
    )
    # Sin rango para el tipo no se inventa uno permisivo: apagaría el criterio
    # en silencio.
    with pytest.raises(ErrorDeConfiguracion, match="falta el rango anatómico de 'codo'"):
        calcular_angulos_de_corrida(corrida, tmp_path / "angulos", otra)


def test_un_salto_imposible_entre_fotogramas_queda_marcado(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    # Umbral bajísimo: el movimiento normal del brazo ya lo supera en la parte
    # rápida del recorrido, así que tiene que marcar algo pero no todo.
    otra = _con_velocidad_maxima(configuracion, 2.0)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", otra)
    codo = resultado.series["codo_izq"]
    rapido = codo.motivos["velocidad_angular"]
    assert rapido.any() and not rapido.all()

    # Con un umbral que nada alcanza, el motivo no marca nada.
    otra = _con_velocidad_maxima(configuracion, 180.0)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", otra)
    assert not resultado.series["codo_izq"].motivos["velocidad_angular"].any()


def test_la_velocidad_no_marca_a_traves_de_un_hueco(tmp_path, configuracion):
    detectado = np.ones(120, bool)
    detectado[40:60] = False
    corrida = _corrida_filtrada(tmp_path, configuracion, detectado=detectado)
    otra = _con_velocidad_maxima(configuracion, 2.0)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", otra)

    codo = resultado.series["codo_izq"]
    # El primer fotograma después del hueco no tiene con qué compararse hacia
    # atrás; el salto que lo cruzaría no es velocidad.
    assert not codo.motivos["velocidad_angular"][40:60].any()
    assert np.isnan(velocidad_angular(codo.grados)[60])


def test_las_series_sobreviven_la_ida_y_vuelta_al_parquet(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)
    leidas = cargar_series_de_angulo(tmp_path / "angulos")

    assert set(leidas) == set(resultado.series)
    for nombre, serie in resultado.series.items():
        # El Parquet guarda float32: la vuelta no es exacta al bit.
        np.testing.assert_allclose(leidas[nombre].grados, serie.grados, rtol=1e-6)
        for motivo in MOTIVOS:
            np.testing.assert_array_equal(leidas[nombre].motivos[motivo], serie.motivos[motivo])


def test_el_resumen_cuenta_los_marcados_sobre_los_angulos_calculados(tmp_path, configuracion):
    detectado = np.ones(120, bool)
    detectado[40:60] = False
    corrida = _corrida_filtrada(tmp_path, configuracion, visibility_codo=0.1, detectado=detectado)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)

    resumen = resultado.resumenes["codo_izq"]
    assert resumen.con_dato == 100
    assert resumen.marcados == 100
    assert resumen.tasa_marcados == pytest.approx(1.0)
    assert resumen.por_motivo["visibility_baja"] == 100


def test_rango_ignora_los_huecos():
    valores = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
    medido = rango(valores)
    assert medido["n"] == 3
    assert medido["minimo"] == pytest.approx(10.0)
    assert medido["maximo"] == pytest.approx(30.0)
    assert medido["p50"] == pytest.approx(20.0)
    assert rango(np.full(3, np.nan)) == {"n": 0}


def test_resumir_no_suma_dos_veces_un_angulo_marcado_por_varios_motivos(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion, visibility_codo=0.1)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)
    serie = resultado.series["codo_izq"]
    # Se agrega a mano un segundo motivo sobre los mismos fotogramas.
    serie.motivos["intercambio_sospechado"][:] = serie.motivos["visibility_baja"]
    resumen = resumir(serie)
    assert resumen.marcados == resumen.con_dato
    assert sum(resumen.por_motivo.values()) > resumen.marcados


def test_la_metadata_de_los_angulos_encadena_con_el_filtrado(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    resultado = calcular_angulos_de_corrida(corrida, tmp_path / "angulos", configuracion)
    metadata = json.loads(resultado.ruta_metadata.read_text(encoding="utf-8"))

    assert metadata["etapa"] == "angulos"
    assert metadata["origen"]["metadata_filtrado"]["etapa"] == "filtrado"
    assert len(metadata["origen"]["sha256_landmarks_filtrados"]) == 64
    parametros = metadata["parametros"]
    assert parametros["coordenadas"] == "pixeles"
    assert parametros["umbral_visibility_reporte"] == (
        configuracion.calidad.umbral_visibility_reporte
    )
    assert parametros["articulaciones"]["codo_izq"] == "hombro_izq-codo_izq-muneca_izq"
    assert metadata["resultado"]["por_articulacion"]["codo_izq"]["rango_grados"]["n"] == 120


def test_sin_umbral_de_visibility_no_se_calculan_angulos(tmp_path, configuracion):
    corrida = _corrida_filtrada(tmp_path, configuracion)
    otra = configuracion.model_copy(
        update={
            "calidad": configuracion.calidad.model_copy(update={"umbral_visibility_reporte": None})
        }
    )
    with pytest.raises(ErrorDeConfiguracion, match="está en null"):
        calcular_angulos_de_corrida(corrida, tmp_path / "angulos", otra)
