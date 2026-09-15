"""Esqueleto del CLI: ayuda, validación de argumentos y aviso de no implementado."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from swimalyzer.cli import SALIDA_ENTRADA_INVALIDA, SALIDA_NO_IMPLEMENTADO, main

CONFIG_DEL_REPO = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture
def video(tmp_path):
    # Archivo vacío: el esqueleto solo verifica que exista.
    archivo = tmp_path / "video.mp4"
    archivo.touch()
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
    assert "--out" in capsys.readouterr().out


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


def test_extraer_con_argumentos_validos_avisa_que_no_esta_implementado(video, tmp_path, capsys):
    salida = tmp_path / "salida"
    codigo = main(["extraer", str(video), "--out", str(salida), "--config", str(CONFIG_DEL_REPO)])
    assert codigo == SALIDA_NO_IMPLEMENTADO
    assert "no está implementada" in capsys.readouterr().err
    assert not salida.exists()
