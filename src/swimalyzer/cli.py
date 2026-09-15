"""Interfaz de línea de comandos: ``swimalyzer <comando> ...``.

Por ahora los subcomandos solo validan argumentos y configuración; la lógica
de cada etapa todavía no está implementada.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from swimalyzer import __version__
from swimalyzer.config import ErrorDeConfiguracion, Recorte, cargar_configuracion

RUTA_CONFIG_POR_DEFECTO = Path("config.yaml")

# Códigos de salida.
SALIDA_NO_IMPLEMENTADO = 1
SALIDA_ENTRADA_INVALIDA = 2


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swimalyzer",
        description="Análisis biomecánico de la brazada de crol a partir de video.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcomandos = parser.add_subparsers(dest="comando", required=True, metavar="COMANDO")

    extraer = subcomandos.add_parser(
        "extraer",
        help="extrae landmarks de pose de un video a Parquet (sin implementar)",
        description=(
            "Corre MediaPipe Pose Landmarker sobre cada fotograma del video y guarda "
            "los landmarks en Parquet junto con la metadata de la corrida. "
            "Todavía no está implementado: solo valida los argumentos."
        ),
    )
    extraer.add_argument("video", type=Path, help="archivo de video a procesar")
    extraer.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="directorio donde se escriben los landmarks y la metadata",
    )
    extraer.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_POR_DEFECTO,
        help="archivo de configuración (por defecto: %(default)s)",
    )
    extraer.add_argument(
        "--recorte",
        type=int,
        nargs=4,
        metavar=("X", "Y", "ANCHO", "ALTO"),
        help="recorte en píxeles del video original; sobrescribe video.recorte de la config",
    )
    extraer.set_defaults(funcion=_comando_extraer)
    return parser


def _error(comando: str, mensaje: str) -> int:
    print(f"swimalyzer {comando}: error: {mensaje}", file=sys.stderr)
    return SALIDA_ENTRADA_INVALIDA


def _comando_extraer(args: argparse.Namespace) -> int:
    if not args.video.is_file():
        return _error("extraer", f"no existe el video: {args.video}")
    if args.out.exists() and not args.out.is_dir():
        return _error("extraer", f"--out existe y no es un directorio: {args.out}")

    try:
        configuracion = cargar_configuracion(args.config)
    except ErrorDeConfiguracion as error:
        return _error("extraer", str(error))

    if args.recorte is not None:
        x, y, ancho, alto = args.recorte
        try:
            recorte = Recorte(x=x, y=y, ancho=ancho, alto=alto)
        except ValidationError as error:
            detalles = "; ".join(f"{d['loc'][0]}: {d['msg']}" for d in error.errors())
            return _error("extraer", f"--recorte inválido ({detalles})")
        video = configuracion.video.model_copy(update={"recorte": recorte})
        configuracion = configuracion.model_copy(update={"video": video})

    print(
        "swimalyzer extraer: la extracción de landmarks todavía no está implementada. "
        f"Argumentos válidos (video={args.video}, out={args.out}, "
        f"recorte={configuracion.video.recorte}).",
        file=sys.stderr,
    )
    return SALIDA_NO_IMPLEMENTADO


def main(argv: list[str] | None = None) -> int:
    args = _construir_parser().parse_args(argv)
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
