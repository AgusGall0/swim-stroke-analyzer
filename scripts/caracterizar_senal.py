"""Informe de caracterización de la señal cruda de una corrida de extracción.

Lee el Parquet de landmarks y su metadata, y escribe números y figuras que son
la evidencia con la que se eligen el umbral de visibility, la frecuencia de
corte y el criterio de intercambios izquierda/derecha. **No elige nada.**

Uso:
    python scripts/caracterizar_senal.py salidas/crol_recortado [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from swimalyzer.pose import landmarks as lm
from swimalyzer.signal.caracterizacion import (
    analisis_de_residuos,
    candidatos_a_intercambio,
    cargar_series,
    descarte_por_umbral,
    espectro,
    frecuencia_de_potencia_acumulada,
    huecos,
    percentiles_de_visibility,
    tramos_continuos,
)
from swimalyzer.viz import caracterizacion as viz

UMBRALES = (0.3, 0.5, 0.7, 0.9)
ORDEN_BUTTERWORTH = 2
#: Margen y separación mínimos, en píxeles, para marcar un candidato a
#: intercambio. Son parámetros del informe, no decisiones del pipeline.
MARGEN_INTERCAMBIO_PX = 20.0
SEPARACION_MINIMA_PX = 20.0


def _tabla(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |"]
    lineas.append("|" + "|".join("---" for _ in encabezados) + "|")
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corrida", type=Path, help="directorio de una corrida de `extraer`")
    parser.add_argument(
        "--out", type=Path, help="dónde escribir el informe (por defecto, en la corrida)"
    )
    args = parser.parse_args(argv)

    destino = args.out or args.corrida / "caracterizacion"
    destino.mkdir(parents=True, exist_ok=True)
    series = cargar_series(args.corrida)
    metadata = json.loads((args.corrida / "metadata.json").read_text(encoding="utf-8"))

    detectado = series.detectado
    tramos = sorted(tramos_continuos(detectado), key=lambda t: t[1] - t[0], reverse=True)
    tramo_mas_largo = tramos[0]
    faltantes = huecos(detectado)

    resumen: dict = {
        "corrida": str(args.corrida),
        "video": metadata["video"],
        "fotogramas": series.cantidad_fotogramas,
        "fotogramas_con_deteccion": int(detectado.sum()),
        "tasa_sin_deteccion": float(1 - detectado.mean()),
        "huecos": [{"inicio": a, "largo": b - a} for a, b in faltantes],
        "tramos_continuos": [{"inicio": a, "largo": b - a} for a, b in tramos],
        "tramo_analizado": {
            "inicio": tramo_mas_largo[0],
            "largo": tramo_mas_largo[1] - tramo_mas_largo[0],
            "duracion_s": (tramo_mas_largo[1] - tramo_mas_largo[0]) / series.fps,
        },
        "visibility": {},
        "descartes": {},
        "espectro": {},
        "residuos": {},
        "intercambios": {},
    }

    for landmark in viz.ORDEN_DE_LANDMARKS:
        nombre = lm.nombre(landmark)
        resumen["visibility"][nombre] = percentiles_de_visibility(series, landmark)
        resumen["descartes"][nombre] = [
            descarte_por_umbral(series, landmark, umbral) for umbral in UMBRALES
        ]

    inicio, fin = tramo_mas_largo
    for landmark in viz.ORDEN_DE_LANDMARKS:
        nombre = lm.nombre(landmark)
        for eje, coordenada in (("x", series.x), ("y", series.y)):
            serie = coordenada[inicio:fin, landmark]
            frecuencias, potencia = espectro(serie, series.fps)
            residuos = analisis_de_residuos(serie, series.fps, orden=ORDEN_BUTTERWORTH)
            resumen["espectro"][f"{nombre}_{eje}"] = {
                "f50_hz": frecuencia_de_potencia_acumulada(frecuencias, potencia, 0.50),
                "f95_hz": frecuencia_de_potencia_acumulada(frecuencias, potencia, 0.95),
                "f99_hz": frecuencia_de_potencia_acumulada(frecuencias, potencia, 0.99),
                "pico_hz": float(frecuencias[1:][int(np.argmax(potencia[1:]))]),
                "rango_px": float(np.ptp(serie)),
            }
            resumen["residuos"][f"{nombre}_{eje}"] = {
                "corte_optimo_hz": residuos.corte_optimo_hz,
                "ruido_estimado_px": residuos.ruido_estimado_px,
            }

    candidatos_por_segmento: dict[str, np.ndarray] = {}
    for nombre, izquierdo, derecho in viz.SEGMENTOS:
        candidatos = candidatos_a_intercambio(
            series,
            izquierdo,
            derecho,
            margen_minimo_px=MARGEN_INTERCAMBIO_PX,
            separacion_minima_px=SEPARACION_MINIMA_PX,
        )
        candidatos_por_segmento[nombre] = candidatos.fotogramas
        separacion = np.linalg.norm(
            np.stack(
                [
                    series.x[detectado, izquierdo] - series.x[detectado, derecho],
                    series.y[detectado, izquierdo] - series.y[detectado, derecho],
                ]
            ),
            axis=0,
        )
        resumen["intercambios"][nombre] = {
            "transiciones_evaluadas": candidatos.transiciones_evaluadas,
            "candidatos": int(candidatos.fotogramas.size),
            "tasa": float(candidatos.fotogramas.size / max(candidatos.transiciones_evaluadas, 1)),
            "margen_mediano_px": (
                float(np.median(candidatos.margen_px)) if candidatos.margen_px.size else None
            ),
            "margen_maximo_px": (
                float(candidatos.margen_px.max()) if candidatos.margen_px.size else None
            ),
            "separacion_mediana_px": float(np.median(separacion)),
            "separacion_p5_px": float(np.percentile(separacion, 5)),
            "cruces_en_x": candidatos.cruces_en_x,
            "fotogramas": candidatos.fotogramas.tolist(),
        }

    figuras = {
        "visibility": viz.figura_visibility(series, destino / "visibility_por_landmark.png"),
        "descartes": viz.figura_descartes(series, destino / "descarte_por_umbral.png"),
        "disponibilidad": viz.figura_disponibilidad(series, destino / "disponibilidad.png"),
        "espectros": viz.figura_espectros(series, tramo_mas_largo, destino / "espectros.png"),
        "residuos": viz.figura_residuos(
            series, tramo_mas_largo, destino / "residuos.png", orden=ORDEN_BUTTERWORTH
        ),
        "intercambios": viz.figura_intercambios(
            series,
            {
                nombre: fotogramas
                for nombre, fotogramas in candidatos_por_segmento.items()
                if nombre in ("codo", "muñeca")
            },
            tramo_mas_largo,
            destino / "intercambios.png",
        ),
    }

    (destino / "caracterizacion.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destino / "informe.md").write_text(_informe(resumen, figuras, series), encoding="utf-8")
    print(f"Informe en {destino / 'informe.md'}")
    for figura in figuras.values():
        print(f"  {figura}")
    return 0


def _informe(resumen: dict, figuras: dict, series) -> str:
    ancho, alto = resumen["video"]["resolucion_inferencia"]
    partes = [
        "# Caracterización de la señal cruda",
        "",
        f"Corrida: `{resumen['corrida']}` · video `{Path(resumen['video']['archivo']).name}` · "
        f"{ancho}×{alto} px a {resumen['video']['fps']:g} fps · "
        f"{resumen['fotogramas']} fotogramas ({resumen['fotogramas'] / series.fps:.1f} s).",
        "",
        "Todo lo que sigue sale de esa corrida. Ningún número está estimado a ojo.",
        "",
        "## 1. Detección",
        "",
        f"- Fotogramas con detección: **{resumen['fotogramas_con_deteccion']}** de "
        f"{resumen['fotogramas']} ({1 - resumen['tasa_sin_deteccion']:.1%}).",
        f"- Tramo continuo más largo: **{resumen['tramo_analizado']['largo']} fotogramas "
        f"({resumen['tramo_analizado']['duracion_s']:.1f} s)**, desde el "
        f"{resumen['tramo_analizado']['inicio']}.",
        f"- Huecos: {len(resumen['huecos'])}, de "
        f"{min(h['largo'] for h in resumen['huecos'])} a "
        f"{max(h['largo'] for h in resumen['huecos'])} fotogramas.",
        "",
        "## 2. Distribución de visibility",
        "",
        "Sobre los fotogramas con detección. Los que no tuvieron detección no aportan",
        "un valor bajo: no aportan valor, y se cuentan aparte.",
        "",
        _tabla(
            ["landmark", "n", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "media"],
            [
                [nombre, str(d["n"])]
                + [f"{d[clave]:.3f}" for clave in ("p5", "p10", "p25", "p50", "p75", "p90", "p95")]
                + [f"{d['media']:.3f}"]
                for nombre, d in resumen["visibility"].items()
            ],
        ),
        "",
        f"![visibility]({figuras['visibility'].name})",
        "",
        "## 3. Qué cuesta cada umbral",
        "",
        "Porcentaje de fotogramas que quedan sin dato usable, sobre los "
        f"{resumen['fotogramas']} del video (incluye los que no tuvieron detección) y,",
        "entre paréntesis, solo sobre los fotogramas con detección.",
        "",
        _tabla(
            ["landmark"] + [f"≥ {u:g}" for u in UMBRALES],
            [
                [nombre]
                + [
                    f"{d['descartados_sobre_todos']:.1%} ({d['descartados_sobre_detectados']:.1%})"
                    for d in descartes
                ]
                for nombre, descartes in resumen["descartes"].items()
            ],
        ),
        "",
        f"![descartes]({figuras['descartes'].name})",
        "",
        f"![disponibilidad]({figuras['disponibilidad'].name})",
        "",
        "## 4. Contenido frecuencial",
        "",
        "Welch sobre el tramo continuo más largo "
        f"({resumen['tramo_analizado']['duracion_s']:.1f} s). "
        "`f95` y `f99` son las frecuencias por debajo de las cuales está el 95 % y el 99 % de la "
        "potencia; `pico` es la componente dominante.",
        "",
        _tabla(
            ["trayectoria", "f50 (Hz)", "f95 (Hz)", "f99 (Hz)", "pico (Hz)", "rango (px)"],
            [
                [
                    clave,
                    f"{d['f50_hz']:.2f}",
                    f"{d['f95_hz']:.2f}",
                    f"{d['f99_hz']:.2f}",
                    f"{d['pico_hz']:.2f}",
                    f"{d['rango_px']:.0f}",
                ]
                for clave, d in resumen["espectro"].items()
            ],
        ),
        "",
        f"![espectros]({figuras['espectros'].name})",
        "",
        "### Análisis de residuos (Winter)",
        "",
        f"Butterworth de orden {ORDEN_BUTTERWORTH} con filtfilt, coordenada x e y por separado.",
        "",
        _tabla(
            ["trayectoria", "corte óptimo (Hz)", "ruido estimado (px)"],
            [
                [clave, f"{d['corte_optimo_hz']:.2f}", f"{d['ruido_estimado_px']:.2f}"]
                for clave, d in resumen["residuos"].items()
            ],
        ),
        "",
        f"![residuos]({figuras['residuos'].name})",
        "",
        "## 5. Candidatos a intercambio izquierda / derecha",
        "",
        f"Se marca una transición entre fotogramas seguidos cuando intercambiar las etiquetas "
        f"explica el movimiento con al menos {MARGEN_INTERCAMBIO_PX:g} px menos de desplazamiento "
        f"total, y los dos landmarks estaban separados más de {SEPARACION_MINIMA_PX:g} px en ambos "
        "fotogramas.",
        "",
        _tabla(
            [
                "par",
                "transiciones",
                "candidatos",
                "tasa",
                "margen mediano (px)",
                "separación mediana (px)",
                "cruces en x",
            ],
            [
                [
                    nombre,
                    str(d["transiciones_evaluadas"]),
                    str(d["candidatos"]),
                    f"{d['tasa']:.1%}",
                    "—" if d["margen_mediano_px"] is None else f"{d['margen_mediano_px']:.0f}",
                    f"{d['separacion_mediana_px']:.0f}",
                    str(d["cruces_en_x"]),
                ]
                for nombre, d in resumen["intercambios"].items()
            ],
        ),
        "",
        f"![intercambios]({figuras['intercambios'].name})",
        "",
        "---",
        "",
        "Este informe no elige el umbral de visibility, la frecuencia de corte ni el criterio de",
        "intercambios. Esas decisiones se toman mirando estos números.",
        "",
    ]
    return "\n".join(partes)


if __name__ == "__main__":
    sys.exit(main())
