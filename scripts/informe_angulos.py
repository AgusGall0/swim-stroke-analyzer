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
)
from swimalyzer.signal.caracterizacion import tramos_continuos
from swimalyzer.viz import angulos as viz
from swimalyzer.viz.angulos import ETIQUETAS_DE_MOTIVO

#: Articulación cuya serie temporal se dibuja: es la que describe el ciclo.
ARTICULACION_DE_REFERENCIA = "codo_izq"

#: Piso anatómico aproximado de cada articulación, en grados de ángulo
#: incluido, para contar cuántas mediciones caen por debajo de lo que el
#: cuerpo puede hacer. Salen del rango de movimiento habitual (flexión máxima
#: de unos 145° en codo y unos 140° en rodilla, que dejan un ángulo incluido
#: de 35° y 40°). Son **parámetros del informe**, no umbrales del pipeline: no
#: filtran nada, solo cuentan.
PISO_ANATOMICO_GRADOS: dict[str, float] = {
    "codo_izq": 35.0,
    "rodilla_izq": 40.0,
    # El hombro llega a juntar el brazo con el tronco: no tiene piso útil.
    "hombro_izq": 0.0,
}


def _tabla(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |"]
    lineas.append("|" + "|".join("---" for _ in encabezados) + "|")
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def _tramo_mas_largo(serie: SerieDeAngulo) -> tuple[int, int]:
    """El tramo continuo con ángulo más largo de la serie."""
    tramos = tramos_continuos(serie.con_dato)
    return max(tramos, key=lambda tramo: tramo[1] - tramo[0])


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
        piso = PISO_ANATOMICO_GRADOS[nombre]
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
            "piso_anatomico_grados": piso,
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
        "eje —y en crol rota— el segmento sale del plano y el ángulo medido es menor que el",
        "real. Con una sola cámara, flexión y escorzo no se distinguen.",
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
        "Un ángulo es tan confiable como su peor componente: hereda las banderas de sus tres",
        "landmarks. Los porcentajes son **sobre los ángulos calculados**, no sobre los",
        f"fotogramas del video. El umbral de visibility de reporte es "
        f"{resumen['umbral_visibility_reporte']:g}.",
        "",
        _tabla(
            ["articulación", "marcados", *(ETIQUETAS_DE_MOTIVO[m] for m in MOTIVOS)],
            [
                [
                    nombre,
                    f"**{datos['marcados']} ({datos['tasa_marcados']:.1%})**",
                    *(
                        f"{datos['por_motivo'][motivo]['fotogramas']} "
                        f"({datos['por_motivo'][motivo]['tasa']:.1%})"
                        for motivo in MOTIVOS
                    ),
                ]
                for nombre, datos in resumen["por_articulacion"].items()
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
        "### Valores por debajo de lo anatómicamente posible",
        "",
        "El piso de referencia es el ángulo incluido que queda con la articulación en flexión",
        "máxima (unos 35° en el codo y 40° en la rodilla). Nada se filtra por esto: se cuenta.",
        "",
        _tabla(
            ["articulación", "piso", "por debajo", "de esos, marcados"],
            [
                [
                    nombre,
                    f"{datos['piso_anatomico_grados']:.0f}°",
                    f"{datos['por_debajo_del_piso']['fotogramas']} "
                    f"({datos['por_debajo_del_piso']['tasa']:.1%})",
                    str(datos["por_debajo_del_piso"]["marcados"]),
                ]
                for nombre, datos in resumen["por_articulacion"].items()
                if datos["piso_anatomico_grados"] > 0
            ],
        ),
        "",
        "## 5. Serie temporal del ángulo de codo",
        "",
        f"Tramo continuo de {tramo['duracion_s']:.1f} s. Abajo, qué fotogramas están marcados",
        "y por qué motivo.",
        "",
        "![serie de codo](serie_codo.png)",
        "",
        "---",
        "",
        "Este informe no elige ningún criterio de segmentación de ciclos: esa sigue siendo una",
        "decisión abierta.",
        "",
    ]
    return "\n".join(partes)


if __name__ == "__main__":
    raise SystemExit(main())
