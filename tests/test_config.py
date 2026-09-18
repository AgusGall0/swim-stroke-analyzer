"""Carga y validación de config.yaml."""

from pathlib import Path

import pytest
import yaml

from swimalyzer.config import Configuracion, ErrorDeConfiguracion, cargar_configuracion

CONFIG_DEL_REPO = Path(__file__).resolve().parent.parent / "config.yaml"

# Lo que sigue sin definir en el config del repo. Hoy está vacío: todas las
# decisiones se tomaron con su informe a la vista. La lista se queda para que
# agregar una decisión abierta traiga su test, y el mecanismo de `null` sigue
# cubierto por `test_exigir_falla_con_un_valor_en_null`.
DECISIONES_ABIERTAS: list[str] = []

#: Valores ya decididos: `exigir` tiene que devolverlos sin protestar.
DECISIONES_TOMADAS = [
    "calidad.umbral_visibility_reporte",
    "calidad.rango_anatomico_grados",
    "calidad.velocidad_angular_maxima_grados_por_fotograma",
    "filtrado.tipo",
    "filtrado.orden",
    "filtrado.frecuencia_corte_hz",
    "filtrado.hueco_maximo_interpolable_fotogramas",
    "lateralidad.metodo_correccion_intercambios",
    "lateralidad.margen_minimo_px",
    "lateralidad.separacion_minima_px",
    "segmentacion.senal",
    "segmentacion.distancia_minima_entre_picos_s",
]


def _datos_del_repo() -> dict:
    return yaml.safe_load(CONFIG_DEL_REPO.read_text(encoding="utf-8"))


def _escribir(datos: dict, carpeta: Path) -> Path:
    archivo = carpeta / "config.yaml"
    archivo.write_text(yaml.safe_dump(datos), encoding="utf-8")
    return archivo


#: Parámetros cuyas claves internas son datos y no esquema: son un mapeo de
#: valores (por ejemplo, un rango por tipo de articulación), así que que falte
#: una clave no es un parámetro faltante. Que estén todas las que hacen falta lo
#: verifica la etapa que las usa, con un mensaje que nombra la que falta.
MAPEOS_DE_VALORES = ["calidad.rango_anatomico_grados"]


def _todos_los_parametros(datos: dict, prefijo: str = "") -> list[str]:
    parametros = []
    for clave, valor in datos.items():
        ruta = f"{prefijo}{clave}"
        parametros.append(ruta)
        if isinstance(valor, dict) and ruta not in MAPEOS_DE_VALORES:
            parametros.extend(_todos_los_parametros(valor, f"{ruta}."))
    return parametros


def test_config_del_repo_carga():
    configuracion = cargar_configuracion(CONFIG_DEL_REPO)
    assert isinstance(configuracion, Configuracion)


def test_ruta_del_modelo_se_resuelve_respecto_de_la_config(tmp_path):
    configuracion = cargar_configuracion(_escribir(_datos_del_repo(), tmp_path))
    assert configuracion.modelo.ruta == tmp_path / "modelos" / "pose_landmarker_heavy.task"


@pytest.mark.parametrize("parametro", _todos_los_parametros(_datos_del_repo()))
def test_falla_si_falta_un_parametro(tmp_path, parametro):
    datos = _datos_del_repo()
    *secciones, clave = parametro.split(".")
    contenedor = datos
    for seccion in secciones:
        contenedor = contenedor[seccion]
    del contenedor[clave]

    with pytest.raises(ErrorDeConfiguracion, match=f"falta el parámetro obligatorio '{parametro}'"):
        cargar_configuracion(_escribir(datos, tmp_path))


def test_falla_si_sobra_un_parametro(tmp_path):
    datos = _datos_del_repo()
    datos["filtrado"]["frecuencia_de_corte"] = 6.0

    with pytest.raises(ErrorDeConfiguracion, match="desconocido 'filtrado.frecuencia_de_corte'"):
        cargar_configuracion(_escribir(datos, tmp_path))


def test_falla_si_un_valor_esta_fuera_de_rango(tmp_path):
    datos = _datos_del_repo()
    datos["deteccion"]["min_tracking_confidence"] = 1.5

    with pytest.raises(ErrorDeConfiguracion, match="deteccion.min_tracking_confidence"):
        cargar_configuracion(_escribir(datos, tmp_path))


def test_falla_si_el_archivo_no_existe(tmp_path):
    with pytest.raises(ErrorDeConfiguracion, match="No existe"):
        cargar_configuracion(tmp_path / "no_existe.yaml")


def test_falla_si_el_archivo_esta_vacio(tmp_path):
    archivo = tmp_path / "config.yaml"
    archivo.write_text("", encoding="utf-8")
    with pytest.raises(ErrorDeConfiguracion, match="mapeo"):
        cargar_configuracion(archivo)


@pytest.mark.parametrize("parametro", DECISIONES_ABIERTAS)
def test_decisiones_abiertas_sin_definir_en_el_repo(parametro):
    configuracion = cargar_configuracion(CONFIG_DEL_REPO)
    with pytest.raises(ErrorDeConfiguracion, match=f"'{parametro}' está en null"):
        configuracion.exigir(parametro)


@pytest.mark.parametrize("parametro", DECISIONES_TOMADAS)
def test_decisiones_ya_tomadas_estan_definidas_en_el_repo(parametro):
    assert cargar_configuracion(CONFIG_DEL_REPO).exigir(parametro) is not None


def test_exigir_falla_con_un_valor_en_null(tmp_path):
    """El freno de las decisiones abiertas: en null, la etapa que lo pida no corre.

    Es lo que evita que un valor provisorio termine en una figura sin que nadie
    recuerde que era provisorio.
    """
    datos = _datos_del_repo()
    datos["segmentacion"]["distancia_minima_entre_picos_s"] = None
    configuracion = cargar_configuracion(_escribir(datos, tmp_path))

    with pytest.raises(
        ErrorDeConfiguracion, match="'segmentacion.distancia_minima_entre_picos_s' está en null"
    ):
        configuracion.exigir("segmentacion.distancia_minima_entre_picos_s")


def test_exigir_devuelve_el_valor_definido():
    configuracion = cargar_configuracion(CONFIG_DEL_REPO)
    assert configuracion.exigir("deteccion.num_poses") == configuracion.deteccion.num_poses


def test_exigir_parametro_inexistente_es_keyerror():
    configuracion = cargar_configuracion(CONFIG_DEL_REPO)
    with pytest.raises(KeyError):
        configuracion.exigir("filtrado.no_existe")
