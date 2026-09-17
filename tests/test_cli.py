"""CLI: ayuda, validación de argumentos y errores de ejecución."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from swimalyzer.cli import SALIDA_ENTRADA_INVALIDA, main

from .conftest import CONFIG_DEL_REPO


@pytest.fixture
def video(tmp_path):
    # Archivo vacío: sirve para los chequeos que no llegan a decodificar nada.
    archivo = tmp_path / "video.mp4"
    archivo.touch()
    return archivo


def config_con_modelo(tmp_path, ruta_modelo: str) -> Path:
    """Copia del config del repo apuntando a otro modelo (existente o no)."""
    datos = yaml.safe_load(CONFIG_DEL_REPO.read_text(encoding="utf-8"))
    datos["modelo"]["ruta"] = ruta_modelo
    archivo = tmp_path / "config.yaml"
    archivo.write_text(yaml.safe_dump(datos, allow_unicode=True), encoding="utf-8")
    return archivo


def test_comando_instalado_responde_a_help():
    # Se busca primero junto al intérprete, por si el entorno virtual no está activado.
    ejecutable = shutil.which("swimalyzer", path=str(Path(sys.executable).parent)) or shutil.which(
        "swimalyzer"
    )
    assert ejecutable is not None, "el comando swimalyzer no está instalado"
    resultado = subprocess.run([ejecutable, "--help"], capture_output=True, text=True, check=False)
    assert resultado.returncode == 0
    assert "extraer" in resultado.stdout


def test_extraer_responde_a_help(capsys):
    with pytest.raises(SystemExit) as salida:
        main(["extraer", "--help"])
    assert salida.value.code == 0
    salida_estandar = capsys.readouterr().out
    assert "--out" in salida_estandar
    assert "--fps" in salida_estandar


def test_extraer_sin_out_es_error_de_uso(video):
    with pytest.raises(SystemExit) as salida:
        main(["extraer", str(video)])
    assert salida.value.code == 2


def test_extraer_video_inexistente(tmp_path, capsys):
    codigo = main(["extraer", str(tmp_path / "no.mp4"), "--out", str(tmp_path / "salida")])
    assert codigo == SALIDA_ENTRADA_INVALIDA
    assert "no existe el video" in capsys.readouterr().err


def test_extraer_recorte_invalido(video, tmp_path, capsys):
    codigo = main(
        [
            "extraer",
            str(video),
            "--out",
            str(tmp_path / "salida"),
            "--config",
            str(CONFIG_DEL_REPO),
            "--recorte",
            "0",
            "0",
            "0",
            "512",
        ]
    )
    assert codigo == SALIDA_ENTRADA_INVALIDA
    assert "--recorte inválido" in capsys.readouterr().err


def test_extraer_sin_modelo_descargado_avisa_como_conseguirlo(video, tmp_path, capsys):
    config = config_con_modelo(tmp_path, "modelos/no_existe.task")
    codigo = main(
        ["extraer", str(video), "--out", str(tmp_path / "salida"), "--config", str(config)]
    )
    assert codigo == SALIDA_ENTRADA_INVALIDA
    error = capsys.readouterr().err
    assert "no está el modelo" in error
    assert "descargar_modelo.py" in error


def test_extraer_video_ilegible_es_error_de_entrada(video, tmp_path, capsys):
    # El modelo existe (archivo de mentira) para llegar al intento de abrir el video.
    modelo = tmp_path / "modelo.task"
    modelo.write_bytes(b"no es un modelo")
    config = config_con_modelo(tmp_path, str(modelo))
    salida = tmp_path / "salida"
    codigo = main(["extraer", str(video), "--out", str(salida), "--config", str(config)])
    assert codigo == SALIDA_ENTRADA_INVALIDA
    assert "no pudo abrir el video" in capsys.readouterr().err


def test_extraer_con_out_que_es_un_archivo(video, tmp_path, capsys):
    ocupado = tmp_path / "ocupado"
    ocupado.touch()
    codigo = main(["extraer", str(video), "--out", str(ocupado), "--config", str(CONFIG_DEL_REPO)])
    assert codigo == SALIDA_ENTRADA_INVALIDA
    assert "no es un directorio" in capsys.readouterr().err


def test_angulos_responde_a_help(capsys):
    with pytest.raises(SystemExit) as salida:
        main(["angulos", "--help"])
    assert salida.value.code == 0
    salida_estandar = capsys.readouterr().out
    assert "--out" in salida_estandar
    assert "píxeles" in salida_estandar


def test_angulos_sin_corrida_filtrada_dice_que_falta_filtrar(tmp_path, capsys):
    codigo = main(["angulos", str(tmp_path), "--config", str(CONFIG_DEL_REPO)])
    assert codigo == SALIDA_ENTRADA_INVALIDA
    assert "swimalyzer filtrar" in capsys.readouterr().err
