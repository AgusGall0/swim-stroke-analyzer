"""El paquete y sus subpaquetes se importan sin errores."""

import importlib

import pytest

import swimalyzer


@pytest.mark.parametrize(
    "modulo",
    [
        "swimalyzer.io",
        "swimalyzer.pose",
        "swimalyzer.pose.landmarks",
        "swimalyzer.signal",
        "swimalyzer.metrics",
        "swimalyzer.viz",
        "swimalyzer.config",
        "swimalyzer.cli",
    ],
)
def test_modulo_importa(modulo):
    importlib.import_module(modulo)


def test_version_instalada():
    assert swimalyzer.__version__ != "desconocida"
