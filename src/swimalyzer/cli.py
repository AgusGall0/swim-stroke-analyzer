"""Interfaz de línea de comandos: ``swimalyzer <comando> ...``.

Cada subcomando valida sus argumentos y la configuración antes de trabajar, y
delega en el paquete: acá no hay lógica de análisis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from swimalyzer import __version__
from swimalyzer.config import Configuracion, ErrorDeConfiguracion, Recorte, cargar_configuracion
from swimalyzer.io.video import ErrorDeVideo
from swimalyzer.pose.extraccion import ResultadoExtraccion, extraer

RUTA_CONFIG_POR_DEFECTO = Path("config.yaml")

# Códigos de salida.
SALIDA_OK = 0
SALIDA_ERROR_DE_EJECUCION = 1
SALIDA_ENTRADA_INVALIDA = 2


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swimalyzer",
        description="Análisis biomecánico de la brazada de crol a partir de video.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcomandos = parser.add_subparsers(dest="comando", required=True, metavar="COMANDO")

    extraer_parser = subcomandos.add_parser(
        "extraer",
        help="extrae landmarks de pose de un video a Parquet",
        description=(
            "Corre MediaPipe Pose Landmarker sobre cada fotograma del video y guarda "
            "los landmarks en Parquet junto con la metadata de la corrida. Los "
            "fotogramas sin detección quedan registrados, no se saltean."
        ),
    )
    extraer_parser.add_argument("video", type=Path, help="archivo de video a procesar")
    extraer_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="directorio donde se escriben los landmarks y la metadata",
    )
    extraer_parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_POR_DEFECTO,
        help="archivo de configuración (por defecto: %(default)s)",
    )
    extraer_parser.add_argument(
        "--recorte",
        type=int,
        nargs=4,
        metavar=("X", "Y", "ANCHO", "ALTO"),
        help="recorte en píxeles del video original; sobrescribe video.recorte de la config",
    )
    extraer_parser.add_argument(
        "--fps",
        type=float,
        metavar="FPS",
        help=(
            "tasa de fotogramas real del video; solo hace falta si el contenedor "
            "no la informa o la informa mal"
        ),
    )
    extraer_parser.set_defaults(funcion=_comando_extraer)
    return parser


def _error(comando: str, mensaje: str) -> int:
    print(f"swimalyzer {comando}: error: {mensaje}", file=sys.stderr)
    return SALIDA_ENTRADA_INVALIDA


def _aplicar_recorte_de_la_linea_de_comandos(
    configuracion: Configuracion, recorte_crudo: list[int]
) -> Configuracion:
    x, y, ancho, alto = recorte_crudo
    recorte = Recorte(x=x, y=y, ancho=ancho, alto=alto)
    video = configuracion.video.model_copy(update={"recorte": recorte})
    return configuracion.model_copy(update={"video": video})


def _informar_progreso(procesados: int, total: int | None) -> None:
    """Avisa cada 30 fotogramas por stderr, para no ensuciar la salida útil."""
    if procesados % 30 and procesados != total:
        return
    de_cuantos = f"/{total}" if total else ""
    final = "\n" if procesados == total else ""
    print(f"\r  fotograma {procesados}{de_cuantos}", end=final, file=sys.stderr, flush=True)


def _resumir(resultado: ResultadoExtraccion) -> None:
    info = resultado.info
    sin_deteccion = len(resultado.fotogramas_sin_deteccion)
    tamano_kb = resultado.ruta_landmarks.stat().st_size / 1024
    print(f"Video: {info.ruta}")
    print(
        f"  {info.ancho}x{info.alto} a {info.fps:g} fps"
        + (" (fps forzado)" if info.fps_fue_forzado else "")
    )
    print(f"Fotogramas procesados: {resultado.fotogramas_procesados}")
    print(
        f"  con detección:    {resultado.fotogramas_con_deteccion}\n"
        f"  sin detección:    {sin_deteccion} ({resultado.tasa_sin_deteccion:.2%})"
    )
    print(f"Filas escritas: {resultado.filas}")
    print(f"Landmarks: {resultado.ruta_landmarks} ({tamano_kb:.1f} KiB)")
    print(f"Metadata:  {resultado.ruta_metadata}")
    print(f"Duración de la extracción: {resultado.duracion_s:.1f} s")


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
        try:
            configuracion = _aplicar_recorte_de_la_linea_de_comandos(configuracion, args.recorte)
        except ValidationError as error:
            detalles = "; ".join(f"{d['loc'][0]}: {d['msg']}" for d in error.errors())
            return _error("extraer", f"--recorte inválido ({detalles})")

    if not configuracion.modelo.ruta.is_file():
        return _error(
            "extraer",
            f"no está el modelo en {configuracion.modelo.ruta}; "
            "descargalo con python scripts/descargar_modelo.py",
        )

    try:
        resultado = extraer(
            args.video,
            args.out,
            configuracion,
            fps_forzado=args.fps,
            progreso=_informar_progreso,
        )
    except ErrorDeVideo as error:
        return _error("extraer", str(error))
    except OSError as error:
        print(f"swimalyzer extraer: error: {error}", file=sys.stderr)
        return SALIDA_ERROR_DE_EJECUCION

    _resumir(resultado)
    return SALIDA_OK


def main(argv: list[str] | None = None) -> int:
    args = _construir_parser().parse_args(argv)
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
