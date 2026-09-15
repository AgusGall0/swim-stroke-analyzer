"""Descarga el modelo Pose Landmarker y verifica su SHA-256.

Lee la URL, el hash esperado y la ruta de destino de ``config.yaml``. Descarga
a un archivo temporal junto al destino y solo lo mueve a la ruta final si el
hash coincide; si no coincide, lo borra y termina con error. Si el modelo ya
existe con el hash correcto, no descarga nada.

Uso:
    python scripts/descargar_modelo.py [--config config.yaml] [--forzar]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

from swimalyzer.config import ErrorDeConfiguracion, cargar_configuracion

RUTA_CONFIG_POR_DEFECTO = Path(__file__).resolve().parent.parent / "config.yaml"
TAMANO_BLOQUE_BYTES = 1024 * 1024
TIEMPO_ESPERA_RED_S = 60


def sha256_de_archivo(ruta: Path) -> str:
    hash_ = hashlib.sha256()
    with ruta.open("rb") as archivo:
        while bloque := archivo.read(TAMANO_BLOQUE_BYTES):
            hash_.update(bloque)
    return hash_.hexdigest()


def descargar(url: str, destino: Path) -> str:
    """Descarga ``url`` en ``destino`` y devuelve el SHA-256 de lo descargado."""
    hash_ = hashlib.sha256()
    with (
        urllib.request.urlopen(url, timeout=TIEMPO_ESPERA_RED_S) as respuesta,
        destino.open("wb") as archivo,
    ):
        while bloque := respuesta.read(TAMANO_BLOQUE_BYTES):
            hash_.update(bloque)
            archivo.write(bloque)
    return hash_.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_POR_DEFECTO,
        help="archivo de configuración (por defecto: el config.yaml del repo)",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="descarga aunque el modelo ya exista con el hash correcto",
    )
    args = parser.parse_args(argv)

    try:
        configuracion = cargar_configuracion(args.config)
    except ErrorDeConfiguracion as error:
        print(error, file=sys.stderr)
        return 2

    destino = configuracion.modelo.ruta
    esperado = configuracion.modelo.sha256

    if destino.exists() and not args.forzar:
        if sha256_de_archivo(destino) == esperado:
            print(f"El modelo ya está en {destino} y su SHA-256 coincide. Nada que hacer.")
            return 0
        print(
            f"{destino} existe pero su SHA-256 no coincide con el esperado; se descarga de nuevo.",
            file=sys.stderr,
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(destino.name + ".parcial")
    print(f"Descargando {configuracion.modelo.url}")
    try:
        obtenido = descargar(configuracion.modelo.url, temporal)
        if obtenido != esperado:
            print(
                "El SHA-256 de la descarga no coincide. Se descarta el archivo.\n"
                f"  esperado: {esperado}\n"
                f"  obtenido: {obtenido}",
                file=sys.stderr,
            )
            return 1
        temporal.replace(destino)
    except (urllib.error.URLError, OSError) as error:
        print(f"Falló la descarga: {error}", file=sys.stderr)
        return 1
    finally:
        temporal.unlink(missing_ok=True)

    print(f"Modelo verificado y guardado en {destino}")
    print(f"SHA-256: {obtenido}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
