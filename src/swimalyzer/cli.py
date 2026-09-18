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
from swimalyzer.metrics.angulos import (
    MOTIVOS,
    NOMBRE_ANGULOS,
    ResultadoAngulos,
    calcular_angulos_de_corrida,
)
from swimalyzer.metrics.ciclos import (
    ErrorDeSegmentacion,
    ResultadoCiclos,
    segmentar_corrida,
)
from swimalyzer.pose.extraccion import NOMBRE_LANDMARKS, ResultadoExtraccion, extraer
from swimalyzer.signal.filtrado import (
    NOMBRE_FILTRADO,
    ErrorDeFiltrado,
    ResultadoFiltrado,
    filtrar_corrida,
)

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

    filtrar_parser = subcomandos.add_parser(
        "filtrar",
        help="interpola huecos cortos y filtra las trayectorias de una corrida",
        description=(
            "Lee los landmarks de una corrida de `extraer`, interpola los huecos cortos, "
            "filtra las trayectorias en píxeles con un Butterworth sin desfase y marca los "
            "fotogramas sospechados de intercambio izquierda/derecha. No descarta nada por "
            "visibility: la columna viaja con los datos."
        ),
    )
    filtrar_parser.add_argument("corrida", type=Path, help="directorio de una corrida de `extraer`")
    filtrar_parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="dónde escribir el resultado (por defecto, la misma corrida)",
    )
    filtrar_parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_POR_DEFECTO,
        help="archivo de configuración (por defecto: %(default)s)",
    )
    filtrar_parser.set_defaults(funcion=_comando_filtrar)

    angulos_parser = subcomandos.add_parser(
        "angulos",
        help="calcula los ángulos articulares del lado cercano a la cámara",
        description=(
            "Lee los landmarks filtrados de una corrida y calcula, sobre coordenadas en "
            "píxeles, los ángulos de codo, hombro y rodilla izquierdos. Cada ángulo hereda "
            "las banderas de sus tres landmarks: queda marcado si a alguno le faltó pasar "
            "por el filtro, se interpoló, quedó sospechado de intercambio o tiene visibility "
            "por debajo del umbral de reporte. Marcar no es descartar."
        ),
    )
    angulos_parser.add_argument("corrida", type=Path, help="directorio de una corrida ya filtrada")
    angulos_parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="dónde escribir el resultado (por defecto, la misma corrida)",
    )
    angulos_parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_POR_DEFECTO,
        help="archivo de configuración (por defecto: %(default)s)",
    )
    angulos_parser.set_defaults(funcion=_comando_angulos)

    ciclos_parser = subcomandos.add_parser(
        "ciclos",
        # El signo de porcentaje va duplicado: argparse interpola la cadena.
        help="segmenta la serie en ciclos de brazada y normaliza cada uno al 0-100 %%",
        description=(
            "Lee los ángulos de una corrida y corta la serie en ciclos de brazada. El corte "
            "es la mano en lo más alto del recobro, medida como el máximo de la altura de la "
            "muñeca sobre el hombro. Un ciclo no cruza un hueco: los cortes se buscan tramo "
            "continuo por tramo continuo. Los ciclos con mediciones marcadas se incluyen; lo "
            "que se informa es qué fracción está marcada en cada fase del ciclo."
        ),
    )
    ciclos_parser.add_argument(
        "corrida", type=Path, help="directorio de una corrida con ángulos ya calculados"
    )
    ciclos_parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="dónde escribir el resultado (por defecto, la misma corrida)",
    )
    ciclos_parser.add_argument(
        "--config",
        type=Path,
        default=RUTA_CONFIG_POR_DEFECTO,
        help="archivo de configuración (por defecto: %(default)s)",
    )
    ciclos_parser.set_defaults(funcion=_comando_ciclos)
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


def _resumir_filtrado(resultado: ResultadoFiltrado) -> None:
    tamano_kb = resultado.ruta_filtrado.stat().st_size / 1024
    con_dato = resultado.muestras_con_dato
    print(f"Muestras: {resultado.muestras} ({con_dato} con dato)")
    print(
        f"  interpoladas: {resultado.muestras_interpoladas} "
        f"({resultado.fotogramas_interpolados} fotogramas)"
    )
    print(
        f"  filtradas:    {resultado.muestras_filtradas}"
        + (
            f" ({resultado.muestras_filtradas / con_dato:.1%} de las que tienen dato)"
            if con_dato
            else ""
        )
    )
    print(
        f"Tramos (contados por landmark): {resultado.tramos_filtrados} filtrados, "
        f"{resultado.tramos_demasiado_cortos} demasiado cortos para filtfilt (sin filtrar)"
    )
    print("Intercambios sospechados por par:")
    for par, datos in resultado.intercambios.items():
        print(
            f"  {par:<28} {datos['fotogramas_marcados']:>4} de "
            f"{datos['transiciones_evaluadas']} transiciones ({datos['tasa']:.1%})"
        )
    print(f"Landmarks filtrados: {resultado.ruta_filtrado} ({tamano_kb:.1f} KiB)")
    print(f"Metadata:            {resultado.ruta_metadata}")


def _comando_filtrar(args: argparse.Namespace) -> int:
    if not (args.corrida / NOMBRE_LANDMARKS).is_file():
        return _error(
            "filtrar",
            f"no hay una corrida de extracción en {args.corrida} (falta {NOMBRE_LANDMARKS})",
        )

    try:
        configuracion = cargar_configuracion(args.config)
    except ErrorDeConfiguracion as error:
        return _error("filtrar", str(error))

    try:
        resultado = filtrar_corrida(args.corrida, args.out or args.corrida, configuracion)
    except (ErrorDeConfiguracion, ErrorDeFiltrado) as error:
        return _error("filtrar", str(error))
    except OSError as error:
        print(f"swimalyzer filtrar: error: {error}", file=sys.stderr)
        return SALIDA_ERROR_DE_EJECUCION

    _resumir_filtrado(resultado)
    return SALIDA_OK


def _resumir_angulos(resultado: ResultadoAngulos) -> None:
    tamano_kb = resultado.ruta_angulos.stat().st_size / 1024
    print(
        f"Fotogramas: {resultado.fotogramas} · visibility mínima para no marcar: "
        f"{resultado.umbral_visibility:g} · salto máximo: "
        f"{resultado.velocidad_maxima:g}°/fotograma"
    )

    for nombre, resumen in resultado.resumenes.items():
        minimo, maximo = resultado.rangos_anatomicos[nombre]
        print(f"\n{nombre}  (rango anatómico {minimo:g}° a {maximo:g}°)")
        print(
            f"  con ángulo: {resumen.con_dato} de {resumen.fotogramas} "
            f"({resumen.tasa_con_dato:.1%})"
        )
        print(f"  marcados:   {resumen.marcados} ({resumen.tasa_marcados:.1%} de los calculados)")
        for motivo in MOTIVOS:
            cantidad = resumen.por_motivo[motivo]
            tasa = cantidad / resumen.con_dato if resumen.con_dato else 0.0
            print(f"    {motivo:<34} {cantidad:>4} ({tasa:>5.1%})")
        for etiqueta, valores in (
            ("todos", resumen.rango),
            ("sin marcar", resumen.rango_sin_marcar),
        ):
            if not valores.get("n"):
                print(f"  rango, {etiqueta:<11} sin datos")
                continue
            print(
                f"  rango, {etiqueta:<11} n={valores['n']:>4}  "
                f"min {valores['minimo']:>5.1f}  mediana {valores['p50']:>5.1f}  "
                f"max {valores['maximo']:>5.1f}"
            )

    print("\nLos motivos no son excluyentes: un ángulo puede estar marcado por varios.")
    print(f"Ángulos:  {resultado.ruta_angulos} ({tamano_kb:.1f} KiB)")
    print(f"Metadata: {resultado.ruta_metadata}")


def _comando_angulos(args: argparse.Namespace) -> int:
    if not (args.corrida / NOMBRE_FILTRADO).is_file():
        return _error(
            "angulos",
            f"no hay una corrida filtrada en {args.corrida} (falta {NOMBRE_FILTRADO}); "
            "corré primero swimalyzer filtrar",
        )

    try:
        configuracion = cargar_configuracion(args.config)
    except ErrorDeConfiguracion as error:
        return _error("angulos", str(error))

    try:
        resultado = calcular_angulos_de_corrida(
            args.corrida, args.out or args.corrida, configuracion
        )
    except (ErrorDeConfiguracion, ErrorDeFiltrado) as error:
        return _error("angulos", str(error))
    except OSError as error:
        print(f"swimalyzer angulos: error: {error}", file=sys.stderr)
        return SALIDA_ERROR_DE_EJECUCION

    _resumir_angulos(resultado)
    return SALIDA_OK


def _resumir_ciclos(resultado: ResultadoCiclos) -> None:
    tamano_kb = resultado.ruta_ciclos.stat().st_size / 1024
    print(
        f"Señal de corte: {resultado.senal.nombre} · evento: {resultado.senal.evento}\n"
        f"Distancia mínima entre cortes: {resultado.distancia_minima_s:g} s · "
        "sin prominencia mínima"
    )

    if not resultado.ciclos:
        print("\nNo se detectó ningún ciclo completo.")
    else:
        duraciones = resultado.duraciones_s
        print(
            f"\nCiclos: {len(resultado.ciclos)} · duración "
            f"{duraciones.min():.2f} a {duraciones.max():.2f} s "
            f"(mediana {resultado.duracion_mediana_s:.2f} s) · "
            f"frecuencia de brazada {resultado.frecuencia_de_brazada_hz:.2f} Hz"
        )
        for ciclo in resultado.ciclos:
            print(
                f"  ciclo {ciclo.numero:>2}  tramo {ciclo.tramo}  "
                f"fotogramas {ciclo.inicio:>4} a {ciclo.fin:>4}  "
                f"{ciclo.duracion_s(resultado.fps):.2f} s"
            )

    for nombre, curva in resultado.curvas.items():
        if not curva.n:
            print(f"\n{nombre}: ningún ciclo con la serie completa")
            continue
        desvio = f"{float(curva.desvio.mean()):.1f}°" if curva.n > 1 else "sin desvío (1 ciclo)"
        print(
            f"\n{nombre}: {curva.n} ciclos promediados · desvío medio {desvio} · "
            f"marcado medio {curva.cobertura_marcada.mean():.1%} de las mediciones"
        )

    print("\nNingún ciclo se descartó por tener mediciones marcadas: la cobertura se informa.")
    print(f"Ciclos:   {resultado.ruta_ciclos} ({tamano_kb:.1f} KiB)")
    print(f"Metadata: {resultado.ruta_metadata}")


def _comando_ciclos(args: argparse.Namespace) -> int:
    if not (args.corrida / NOMBRE_ANGULOS).is_file():
        return _error(
            "ciclos",
            f"no hay ángulos calculados en {args.corrida} (falta {NOMBRE_ANGULOS}); "
            "corré primero swimalyzer angulos",
        )

    try:
        configuracion = cargar_configuracion(args.config)
    except ErrorDeConfiguracion as error:
        return _error("ciclos", str(error))

    try:
        resultado = segmentar_corrida(args.corrida, args.out or args.corrida, configuracion)
    except (ErrorDeConfiguracion, ErrorDeFiltrado, ErrorDeSegmentacion) as error:
        return _error("ciclos", str(error))
    except OSError as error:
        print(f"swimalyzer ciclos: error: {error}", file=sys.stderr)
        return SALIDA_ERROR_DE_EJECUCION

    _resumir_ciclos(resultado)
    return SALIDA_OK


def main(argv: list[str] | None = None) -> int:
    args = _construir_parser().parse_args(argv)
    return args.funcion(args)


if __name__ == "__main__":
    sys.exit(main())
