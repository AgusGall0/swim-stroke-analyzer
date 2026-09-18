"""Informe de los ángulos articulares de una corrida.

Lee el Parquet de ángulos ya persistido (`swimalyzer angulos`) y escribe
cuántos ángulos hay, qué porcentaje queda marcado y por qué motivo, y qué
rango de valores toma cada articulación. **No decide nada**: los ángulos
marcados no se descartan acá ni en ningún lado.

Uso:
    python scripts/informe_angulos.py salidas/final [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swimalyzer.metrics.angulos import (
    MOTIVOS,
    NOMBRE_METADATA_ANGULOS,
    SerieDeAngulo,
    cargar_series_de_angulo,
    rango,
    resumir,
    salto_adyacente,
    velocidad_angular,
)
from swimalyzer.signal.caracterizacion import tramos_continuos
from swimalyzer.viz import angulos as viz
from swimalyzer.viz.angulos import ETIQUETAS_DE_MOTIVO

#: Articulación cuya serie temporal se dibuja: es la que describe el ciclo.
ARTICULACION_DE_REFERENCIA = "codo_izq"

#: Umbrales de velocidad angular, en grados por fotograma, para los que el
#: informe calcula cuánto marcaría cada uno. Son el barrido del que sale el
#: número, no el número: elegirlo es una decisión abierta.
UMBRALES_DE_VELOCIDAD: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0)

#: Frecuencia de brazada del material, en Hz: la mediana de la duración de los
#: ciclos ya segmentados (1.83 s sobre 7 ciclos, ver el informe de ciclos). Se
#: usa solo para calcular con qué velocidad se movería el ángulo si la brazada
#: fuera una oscilación sinusoidal, que es la referencia contra la cual se lee
#: la velocidad medida. Sobre otro material hay que volver a mirarla.
#:
#: Antes acá decía 0.47 Hz, que era el pico de la PSD del informe de
#: caracterización. Ese espectro usa ventanas de 8 s, o sea que su resolución es
#: de 0.234 Hz y 0.47 era el bin 2: el número no distinguía 0.47 de 0.55. La
#: medición por segmentación es la fina.
FRECUENCIA_DE_BRAZADA_HZ = 0.55

#: Percentiles de la distribución de velocidad angular que van al informe.
PERCENTILES_DE_VELOCIDAD: tuple[float, ...] = (50, 75, 90, 95, 99)


def _tabla(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |"]
    lineas.append("|" + "|".join("---" for _ in encabezados) + "|")
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def _tramo_mas_largo(serie: SerieDeAngulo) -> tuple[int, int]:
    """El tramo continuo con ángulo más largo de la serie."""
    tramos = tramos_continuos(serie.con_dato)
    return max(tramos, key=lambda tramo: tramo[1] - tramo[0])


def _resumir_velocidad(serie: SerieDeAngulo, fps: float, piso: float) -> dict:
    """Distribución del salto entre fotogramas y qué marcaría cada umbral."""
    velocidad = velocidad_angular(serie.grados)
    valores = velocidad[~np.isnan(velocidad)]
    con_angulo = serie.grados[serie.con_dato]
    excursion = float(np.ptp(np.percentile(con_angulo, (5, 95)))) if con_angulo.size else 0.0
    salto = salto_adyacente(velocidad)
    con_dato = serie.con_dato
    with np.errstate(invalid="ignore"):
        fuera_del_piso = con_dato & (serie.grados < piso) if piso > 0 else np.zeros_like(con_dato)

    percentiles = dict(
        zip(
            (f"p{p:g}" for p in PERCENTILES_DE_VELOCIDAD),
            (float(v) for v in np.percentile(valores, PERCENTILES_DE_VELOCIDAD)),
            strict=True,
        )
    )
    costos = []
    for umbral in UMBRALES_DE_VELOCIDAD:
        with np.errstate(invalid="ignore"):
            marcaria = con_dato & (salto >= umbral)
        costos.append(
            {
                "umbral_grados_por_fotograma": umbral,
                "grados_por_segundo": umbral * fps,
                "fotogramas": int(marcaria.sum()),
                "tasa": float(marcaria.sum() / max(int(con_dato.sum()), 1)),
                # Cuántos de los valores anatómicamente imposibles alcanzaría a
                # señalar este umbral: los dos motivos no se solapan del todo.
                "fuera_del_piso_capturados": int((marcaria & fuera_del_piso).sum()),
            }
        )
    return {
        "transiciones": int(valores.size),
        "media": float(valores.mean()),
        # Con qué velocidad se movería este ángulo si la brazada fuera una
        # sinusoide de la excursión medida: 2*pi*f*amplitud, pasado a grados por
        # fotograma. No es una cota, es una referencia para leer los números.
        "pico_sinusoidal_grados_por_fotograma": float(
            2 * np.pi * FRECUENCIA_DE_BRAZADA_HZ * excursion / 2 / fps
        ),
        "excursion_grados": float(excursion),
        "maximo": float(valores.max()),
        "percentiles": percentiles,
        "fuera_del_piso": int(fuera_del_piso.sum()),
        "costo_por_umbral": costos,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corrida", type=Path, help="directorio de una corrida con ángulos")
    parser.add_argument(
        "--out", type=Path, help="dónde escribir el informe (por defecto, en la corrida)"
    )
    args = parser.parse_args(argv)

    destino = args.out or args.corrida / "angulos"
    destino.mkdir(parents=True, exist_ok=True)

    series = cargar_series_de_angulo(args.corrida)
    metadata = json.loads((args.corrida / NOMBRE_METADATA_ANGULOS).read_text(encoding="utf-8"))
    video = metadata["origen"]["metadata_filtrado"]["origen"]["metadata_extraccion"]["video"]
    fps = float(video["fps"])

    referencia = series[ARTICULACION_DE_REFERENCIA]
    tramo = _tramo_mas_largo(referencia)
    resumenes = {nombre: resumir(serie) for nombre, serie in series.items()}

    resumen: dict = {
        "corrida": str(args.corrida),
        "video": video,
        "umbral_visibility_reporte": metadata["parametros"]["umbral_visibility_reporte"],
        "velocidad_maxima": metadata["parametros"]["velocidad_angular_maxima_grados_por_fotograma"],
        "articulaciones": metadata["parametros"]["articulaciones"],
        "tramo_analizado": {
            "articulacion": ARTICULACION_DE_REFERENCIA,
            "inicio": tramo[0],
            "largo": tramo[1] - tramo[0],
            "duracion_s": (tramo[1] - tramo[0]) / fps,
        },
        "por_articulacion": {},
    }
    for nombre, serie in series.items():
        detalle = resumenes[nombre]
        rango_anatomico = metadata["parametros"]["rango_anatomico_grados"][nombre]
        piso = float(rango_anatomico["minimo"])
        with np.errstate(invalid="ignore"):
            debajo_del_piso = serie.con_dato & (serie.grados < piso)
        inicio, fin = tramo
        resumen["por_articulacion"][nombre] = {
            "fotogramas": detalle.fotogramas,
            "con_dato": detalle.con_dato,
            "sin_dato": detalle.fotogramas - detalle.con_dato,
            "marcados": detalle.marcados,
            "tasa_marcados": detalle.tasa_marcados,
            "por_motivo": {
                motivo: {
                    "fotogramas": cantidad,
                    "tasa": cantidad / detalle.con_dato if detalle.con_dato else 0.0,
                }
                for motivo, cantidad in detalle.por_motivo.items()
            },
            "rango_grados": detalle.rango,
            "rango_grados_sin_marcar": detalle.rango_sin_marcar,
            "rango_grados_en_el_tramo": rango(serie.grados[inicio:fin]),
            "velocidad_angular": _resumir_velocidad(serie, fps, piso),
            "rango_anatomico_grados": rango_anatomico,
            "por_debajo_del_piso": {
                "fotogramas": int(debajo_del_piso.sum()),
                "tasa": (
                    float(debajo_del_piso.sum() / detalle.con_dato) if detalle.con_dato else 0.0
                ),
                "marcados": int((debajo_del_piso & serie.marcado).sum()),
            },
        }

    figuras = {
        "serie_codo": viz.figura_serie(referencia, fps, tramo, destino / "serie_codo.png"),
        "rangos": viz.figura_rangos(series, destino / "rangos.png"),
        "velocidad": viz.figura_velocidad_angular(
            series, fps, UMBRALES_DE_VELOCIDAD, destino / "velocidad_angular.png"
        ),
    }

    (destino / "angulos.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destino / "informe.md").write_text(_informe(resumen), encoding="utf-8")
    print(f"Informe en {destino / 'informe.md'}")
    for figura in figuras.values():
        print(f"  {figura}")
    return 0


def _informe(resumen: dict) -> str:
    ancho, alto = resumen["video"]["resolucion_inferencia"]
    tramo = resumen["tramo_analizado"]
    velocidad_del_codo = resumen["por_articulacion"][ARTICULACION_DE_REFERENCIA][
        "velocidad_angular"
    ]
    partes = [
        "# Ángulos articulares del lado cercano",
        "",
        f"Corrida: `{resumen['corrida']}` · video `{Path(resumen['video']['archivo']).name}` · "
        f"{ancho}×{alto} px a {resumen['video']['fps']:g} fps.",
        "",
        "Todo sale del Parquet de ángulos de esa corrida. Ningún número está estimado a ojo.",
        "",
        "## 1. Qué se midió",
        "",
        "Ángulo incluido en el vértice, en grados, sobre coordenadas en píxeles: 180° es el",
        "segmento extendido y el valor baja a medida que la articulación se cierra. Solo el",
        "lado izquierdo, que es el cercano a la cámara.",
        "",
        _tabla(
            ["articulación", "landmarks (proximal-vértice-distal)"],
            [
                [nombre, f"`{descripcion}`"]
                for nombre, descripcion in resumen["articulaciones"].items()
            ],
        ),
        "",
        "Es un ángulo **proyectado** en el plano de la imagen. Con el cuerpo rotado sobre su",
        "eje —y en crol rota— el segmento sale del plano y el ángulo medido difiere del real:",
        "puede quedar por debajo o por encima, porque dos segmentos casi alineados con el eje",
        "de la cámara proyectan a casi 180°. Con una sola cámara, flexión y escorzo no se",
        "distinguen.",
        "",
        "## 2. Cobertura",
        "",
        f"De {next(iter(resumen['por_articulacion'].values()))['fotogramas']} fotogramas del",
        "video, cuántos tienen los tres landmarks y por lo tanto dan un ángulo. Los que no,",
        "quedan en `NaN`: el hueco se registra.",
        "",
        _tabla(
            ["articulación", "con ángulo", "sin ángulo"],
            [
                [
                    nombre,
                    f"{datos['con_dato']} ({datos['con_dato'] / datos['fotogramas']:.1%})",
                    f"{datos['sin_dato']} ({datos['sin_dato'] / datos['fotogramas']:.1%})",
                ]
                for nombre, datos in resumen["por_articulacion"].items()
            ],
        ),
        "",
        f"Tramo continuo de ángulo más largo ({tramo['articulacion']}): "
        f"**{tramo['largo']} fotogramas ({tramo['duracion_s']:.1f} s)**, desde el "
        f"{tramo['inicio']}.",
        "",
        "## 3. Qué porcentaje queda marcado, y por qué",
        "",
        "Cuatro motivos se heredan de los landmarks —un ángulo es tan confiable como su peor",
        "componente— y dos miran el ángulo ya calculado. Los porcentajes son **sobre los",
        "ángulos calculados**, no sobre los fotogramas del video. El umbral de visibility de",
        f"reporte es {resumen['umbral_visibility_reporte']:g} y el salto máximo, "
        f"{resumen['velocidad_maxima']:g}° por fotograma.",
        "",
        _tabla(
            ["motivo", *resumen["por_articulacion"]],
            [
                [
                    ETIQUETAS_DE_MOTIVO[motivo],
                    *(
                        f"{datos['por_motivo'][motivo]['fotogramas']} "
                        f"({datos['por_motivo'][motivo]['tasa']:.1%})"
                        for datos in resumen["por_articulacion"].values()
                    ),
                ]
                for motivo in MOTIVOS
            ]
            + [
                [
                    "**marcado por al menos uno**",
                    *(
                        f"**{datos['marcados']} ({datos['tasa_marcados']:.1%})**"
                        for datos in resumen["por_articulacion"].values()
                    ),
                ]
            ],
        ),
        "",
        "Los motivos no son excluyentes: un mismo ángulo puede estar marcado por varios, así",
        "que la suma de las columnas puede pasar el total de marcados.",
        "",
        "## 4. Rango de valores",
        "",
        "Qué valores toma cada ángulo, para ver si son anatómicamente posibles. Se muestran",
        "por separado todos los ángulos y solo los que no tienen ninguna bandera.",
        "",
        _tabla(
            ["articulación", "conjunto", "n", "mín", "p5", "mediana", "p95", "máx"],
            [
                [
                    nombre,
                    etiqueta,
                    str(valores["n"]),
                    f"{valores['minimo']:.1f}",
                    f"{valores['p5']:.1f}",
                    f"{valores['p50']:.1f}",
                    f"{valores['p95']:.1f}",
                    f"{valores['maximo']:.1f}",
                ]
                for nombre, datos in resumen["por_articulacion"].items()
                for etiqueta, valores in (
                    ("todos", datos["rango_grados"]),
                    ("sin marcar", datos["rango_grados_sin_marcar"]),
                )
                if valores.get("n")
            ],
        ),
        "",
        "![rangos](rangos.png)",
        "",
        "### Valores fuera del rango anatómico",
        "",
        "El rango sale de `config.yaml` y es el que la articulación puede recorrer según los",
        "valores de referencia de goniometría (AAOS; Norkin & White), convertidos a ángulo",
        "incluido con `180 − flexión`. Nada se filtra por esto: se marca y se cuenta.",
        "",
        _tabla(
            ["articulación", "rango", "por debajo del mínimo", "de esos, marcados por otro motivo"],
            [
                [
                    nombre,
                    f"{datos['rango_anatomico_grados']['minimo']:.0f}° a "
                    f"{datos['rango_anatomico_grados']['maximo']:.0f}°",
                    f"{datos['por_debajo_del_piso']['fotogramas']} "
                    f"({datos['por_debajo_del_piso']['tasa']:.1%})",
                    str(datos["por_debajo_del_piso"]["marcados"]),
                ]
                for nombre, datos in resumen["por_articulacion"].items()
            ],
        ),
        "",
        "Un valor fuera del rango **no prueba que el landmark esté mal**. La proyección puede",
        "achicar el ángulo tanto como agrandarlo: con el segmento apuntando a la cámara, un",
        "codo realmente flexionado a 90° puede proyectar cualquier cosa. Puede ser un landmark",
        "mal estimado o escorzo extremo, y con una sola cámara no se distingue cuál. Lo que sí",
        "se sabe es que la medición no representa a la articulación, y eso alcanza para",
        "marcarla. Por eso el motivo se llama `fuera_de_rango_en_el_plano_medido` y no algo",
        "que culpe al landmark.",
        "",
        "## 5. Velocidad angular",
        "",
        "Cuánto cambia cada ángulo entre fotogramas consecutivos, en grados por fotograma. Es",
        "la evidencia para elegir un umbral de salto: un artefacto de un fotograma entra y sale",
        "con dos saltos grandes, mientras que el movimiento real está acotado por la frecuencia",
        "de corte del filtro. El primer fotograma de cada tramo no tiene con qué compararse.",
        "",
        f"**La mediana del codo ({velocidad_del_codo['percentiles']['p50']:.1f}° por fotograma) ya",
        "supera el pico que implica la brazada "
        f"({velocidad_del_codo['pico_sinusoidal_grados_por_fotograma']:.1f}° por fotograma: una",
        f"oscilación sinusoidal a {FRECUENCIA_DE_BRAZADA_HZ:g} Hz con la excursión medida de "
        f"{velocidad_del_codo['excursion_grados']:.0f}°).** El umbral elegido marca lo grosero:",
        "que un ángulo no quede marcado por velocidad **no significa que esté limpio**. La mitad",
        "de la serie se mueve más rápido de lo que la brazada explica, y eso es ruido que el",
        "filtro dejó pasar.",
        "",
        _tabla(
            ["articulación", "n", "media", *(f"p{p:g}" for p in PERCENTILES_DE_VELOCIDAD), "máx"],
            [
                [
                    nombre,
                    str(datos["velocidad_angular"]["transiciones"]),
                    f"{datos['velocidad_angular']['media']:.1f}",
                    *(
                        f"{datos['velocidad_angular']['percentiles'][f'p{p:g}']:.1f}"
                        for p in PERCENTILES_DE_VELOCIDAD
                    ),
                    f"{datos['velocidad_angular']['maximo']:.1f}",
                ]
                for nombre, datos in resumen["por_articulacion"].items()
            ],
        ),
        "",
        "Qué marcaría cada umbral, contando el fotograma cuando el salto que entra **o** el que",
        "sale lo supera. La última columna dice cuántos de los ángulos por debajo del mínimo",
        "anatómico alcanzaría a señalar: los dos criterios no se solapan del todo.",
        "",
        _tabla(
            [
                "umbral (°/fotograma)",
                "equivale a (°/s)",
                *(f"{nombre} marcados" for nombre in resumen["por_articulacion"]),
                "codo: fuera de rango capturados",
            ],
            [
                [
                    f"{umbral:g}",
                    f"{umbral * resumen['video']['fps']:.0f}",
                    *(
                        f"{costo['fotogramas']} ({costo['tasa']:.1%})"
                        for costo in (
                            datos["velocidad_angular"]["costo_por_umbral"][indice]
                            for datos in resumen["por_articulacion"].values()
                        )
                    ),
                    f"{velocidad_del_codo['costo_por_umbral'][indice]['fuera_del_piso_capturados']}"
                    f" de {velocidad_del_codo['fuera_del_piso']}",
                ]
                for indice, umbral in enumerate(UMBRALES_DE_VELOCIDAD)
            ],
        ),
        "",
        "![velocidad angular](velocidad_angular.png)",
        "",
        "## 6. Serie temporal del ángulo de codo",
        "",
        f"Tramo continuo de {tramo['duracion_s']:.1f} s. Abajo, qué fotogramas están marcados",
        "y por qué motivo.",
        "",
        "![serie de codo](serie_codo.png)",
        "",
        "---",
        "",
        "Este informe no segmenta ciclos: eso lo hace `swimalyzer ciclos` y se documenta en",
        "`ciclos/informe.md`. La frecuencia de brazada que se usa acá como referencia sale de",
        "esa segmentación.",
        "",
    ]
    return "\n".join(partes)


if __name__ == "__main__":
    raise SystemExit(main())
