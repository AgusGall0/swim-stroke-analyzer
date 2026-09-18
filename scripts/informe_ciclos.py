"""Informe de la segmentación en ciclos de brazada de una corrida.

Lee el Parquet de ciclos ya persistido (`swimalyzer ciclos`) y escribe dónde
cayó cada corte, cuánto dura cada ciclo, y la curva media del ángulo de codo
normalizada al 0-100 % del ciclo con su cobertura. **No decide nada**: los
ciclos con mediciones marcadas no se descartan acá ni en ningún lado.

Uso:
    python scripts/informe_ciclos.py salidas/final [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swimalyzer.metrics.angulos import cargar_series_de_angulo
from swimalyzer.metrics.ciclos import (
    ARTICULACION_DE_REFERENCIA,
    NOMBRE_METADATA_CICLOS,
    SENALES_DE_SEGMENTACION,
    Ciclo,
    construir_senal,
    curva_media,
)
from swimalyzer.signal.filtrado import cargar_filtrado
from swimalyzer.viz import ciclos as viz


def _tabla(encabezados: list[str], filas: list[list[str]]) -> str:
    lineas = ["| " + " | ".join(encabezados) + " |"]
    lineas.append("|" + "|".join("---" for _ in encabezados) + "|")
    lineas += ["| " + " | ".join(fila) + " |" for fila in filas]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corrida", type=Path, help="directorio de una corrida con ciclos")
    parser.add_argument(
        "--out", type=Path, help="dónde escribir el informe (por defecto, en la corrida)"
    )
    args = parser.parse_args(argv)

    destino = args.out or args.corrida / "ciclos"
    destino.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((args.corrida / NOMBRE_METADATA_CICLOS).read_text(encoding="utf-8"))
    parametros = metadata["parametros"]
    resultado = metadata["resultado"]
    video = metadata["origen"]["metadata_angulos"]["origen"]["metadata_filtrado"]["origen"][
        "metadata_extraccion"
    ]["video"]
    fps = float(resultado["fps"])

    # Las curvas se rearman desde los datos persistidos: el informe dibuja lo
    # que está guardado, no un cálculo hecho al paso.
    series = cargar_series_de_angulo(args.corrida)
    filtradas = cargar_filtrado(args.corrida)
    senal = construir_senal(filtradas, parametros["senal"])
    ciclos = [
        Ciclo(
            numero=datos["numero"],
            tramo=datos["tramo"],
            inicio=datos["frame_inicio"],
            fin=datos["frame_fin"],
        )
        for datos in resultado["ciclos"]
    ]
    disponible = ~np.isnan(senal) & series[ARTICULACION_DE_REFERENCIA].con_dato
    curvas = {nombre: curva_media(serie, ciclos) for nombre, serie in series.items()}
    referencia = curvas[ARTICULACION_DE_REFERENCIA]

    viz.figura_curva_media(referencia, destino / "curva_media_codo.png")
    viz.figura_segmentacion(
        senal,
        disponible,
        ciclos,
        fps,
        SENALES_DE_SEGMENTACION[parametros["senal"]],
        destino / "segmentacion.png",
    )

    # El ángulo de codo en cada instante de corte. Si el corte fuera el mismo
    # punto del movimiento en todos los ciclos, el codo tendría que estar más o
    # menos igual en todos: la dispersión de estos valores dice cuánto no lo
    # está, y es lo que explica el salto de la curva media entre el 0 % y el
    # 100 %, que son el mismo evento medido sobre conjuntos de cortes distintos.
    cortes = sorted({ciclo.inicio for ciclo in ciclos} | {ciclo.fin for ciclo in ciclos})
    angulo_en_corte = series[ARTICULACION_DE_REFERENCIA].grados[cortes]

    resumen = {
        "corrida": str(args.corrida),
        "video": video,
        "parametros": parametros,
        "resultado": resultado,
        "angulo_de_codo_en_el_corte": {
            "cortes": cortes,
            "grados": [float(valor) for valor in angulo_en_corte],
            "minimo": float(np.nanmin(angulo_en_corte)),
            "mediana": float(np.nanmedian(angulo_en_corte)),
            "maximo": float(np.nanmax(angulo_en_corte)),
            "desvio": float(np.nanstd(angulo_en_corte, ddof=1)),
        },
        "por_articulacion": {
            nombre: {
                "ciclos": curva.n,
                "desvio_medio": float(np.nanmean(curva.desvio)) if curva.n > 1 else None,
                "desvio_maximo": float(np.nanmax(curva.desvio)) if curva.n > 1 else None,
                "cobertura_marcada_media": float(curva.cobertura_marcada.mean()),
                "cobertura_marcada_maxima": float(curva.cobertura_marcada.max()),
                "excursion_de_la_media": float(curva.media.max() - curva.media.min()),
            }
            for nombre, curva in curvas.items()
        },
    }
    (destino / "ciclos.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destino / "informe.md").write_text(_informe(resumen), encoding="utf-8")
    print(f"Informe escrito en {destino}")
    return 0


def _informe(resumen: dict) -> str:
    ancho, alto = resumen["video"]["resolucion_inferencia"]
    parametros = resumen["parametros"]
    resultado = resumen["resultado"]
    duracion = resultado["duracion_de_ciclo_s"]
    codo = resumen["por_articulacion"][ARTICULACION_DE_REFERENCIA]
    corte_codo = resumen["angulo_de_codo_en_el_corte"]
    tramos_con_ciclos = sorted({ciclo["tramo"] for ciclo in resultado["ciclos"]})
    por_tramo = {
        tramo["numero"]: tramo
        for tramo in resultado["tramos_continuos"]
        if tramo["numero"] in tramos_con_ciclos
    }

    partes = [
        "# Ciclos de brazada y curva media del ángulo de codo",
        "",
        f"Corrida: `{resumen['corrida']}` · video `{Path(resumen['video']['archivo']).name}` · "
        f"{ancho}×{alto} px a {resumen['video']['fps']:g} fps.",
        "",
        "Todo sale del Parquet de ciclos de esa corrida. Ningún número está estimado a ojo.",
        "",
        "## 1. Dónde se corta el ciclo",
        "",
        f"El corte es **{parametros['evento_de_corte']}**, medido como el máximo de "
        f"`{parametros['senal']}`: {parametros['senal_descripcion']}.",
        "",
        "La señal es **relativa al hombro** porque el nadador se desplaza dentro del encuadre,",
        "y una coordenada absoluta mezclaría ese desplazamiento con el movimiento del brazo. Es",
        "**vertical** porque es la componente que separa el recobro —brazo arriba— del resto del",
        "ciclo. Y el orden de la resta importa: en coordenadas de imagen la `y` crece hacia",
        "abajo, así que la altura sobre el hombro es `y_hombro − y_muñeca`. Al revés, el máximo",
        "sería la mano más profunda del tirón, que es otro evento y da otros ciclos.",
        "",
        f"Distancia mínima entre cortes: **{parametros['distancia_minima_entre_picos_s']:g} s**.",
        "No hay prominencia mínima: en el barrido de 0 a 20 px no cambió ningún pico, así que",
        "sería un parámetro inerte.",
        "",
        "**Un ciclo no cruza un hueco.** Los cortes se buscan tramo continuo por tramo continuo:",
        "un intervalo que abarque un hueco no es un ciclo, es dos trozos con un tiempo",
        "indeterminado en el medio.",
        "",
        "![segmentación](segmentacion.png)",
        "",
        "## 2. Los ciclos que salieron",
        "",
        f"**{resultado['ciclos_detectados']} ciclos**, de "
        f"{len(tramos_con_ciclos)} tramos continuos. Duración de "
        f"{duracion['minimo']:.2f} a {duracion['maximo']:.2f} s, mediana "
        f"**{duracion['mediana']:.2f} s**, o sea una frecuencia de brazada de "
        f"**{resultado['frecuencia_de_brazada_hz']:.2f} Hz**.",
        "",
        _tabla(
            ["ciclo", "tramo", "fotogramas", "duración (s)"],
            [
                [
                    str(ciclo["numero"]),
                    str(ciclo["tramo"]),
                    f"{ciclo['frame_inicio']} a {ciclo['frame_fin']}",
                    f"{ciclo['duracion_s']:.2f}",
                ]
                for ciclo in resultado["ciclos"]
            ],
        ),
        "",
        "Los tramos que aportaron ciclos:",
        "",
        _tabla(
            ["tramo", "fotogramas", "duración (s)", "ciclos"],
            [
                [
                    str(numero),
                    f"{tramo['frame_inicio']} a {tramo['frame_fin']}",
                    f"{tramo['fotogramas'] / resumen['video']['fps']:.1f}",
                    str(sum(1 for c in resultado["ciclos"] if c["tramo"] == numero)),
                ]
                for numero, tramo in por_tramo.items()
            ],
        ),
        "",
        "La numeración de tramos cuenta **todos** los tramos continuos de la corrida, también",
        "los que no dieron ningún ciclo: por eso los números saltan. La lista completa está en",
        "`metadata_ciclos.json`.",
        "",
        "### Sobre la frecuencia de brazada",
        "",
        f"Los {resultado['frecuencia_de_brazada_hz']:.2f} Hz de acá salen de medir la duración de",
        "estos ciclos. El informe de caracterización reportaba 0.47 Hz como pico de la PSD, pero",
        "ese número venía de un espectro de Welch con ventanas de 8 s, o sea con una resolución",
        "de 0.234 Hz: 0.47 era el bin 2, y no distinguía 0.47 de 0.55. La medición por",
        "segmentación es la fina, y es la que vale.",
        "",
        "## 3. Curva media del ángulo de codo",
        "",
        "Cada ciclo se reinterpola linealmente sobre una malla de 0 a 100 % de su propia",
        "duración. Normalizar así es lo que permite promediar ciclos de distinta duración: el",
        "promedio compara fases equivalentes y no fotogramas equivalentes. El 0 % y el 100 % son",
        "el mismo evento en dos repeticiones consecutivas.",
        "",
        f"**{codo['ciclos']} ciclos promediados.** Desvío estándar medio a lo largo del ciclo: "
        f"**{codo['desvio_medio']:.1f}°**, con un máximo de {codo['desvio_maximo']:.1f}°. La",
        f"curva media recorre {codo['excursion_de_la_media']:.0f}° entre su mínimo y su máximo.",
        "",
        "![curva media](curva_media_codo.png)",
        "",
        "### Qué se descartó: nada",
        "",
        "Ningún ciclo se sacó por tener mediciones marcadas. Con "
        f"{codo['ciclos']} ciclos, descartar los que tienen alguna medición marcada deja la",
        "muestra en nada y, peor, esconde el problema: la figura saldría limpia porque se le",
        "sacó lo sucio, no porque el dato lo sea.",
        "",
        "En su lugar la cobertura viaja con la curva. En promedio a lo largo del ciclo, el",
        f"**{codo['cobertura_marcada_media']:.1%}** de las mediciones que se promedian está",
        f"marcado por algún motivo, y en la peor fase llega al "
        f"**{codo['cobertura_marcada_maxima']:.1%}**. El panel de abajo de la figura lo muestra",
        "fase por fase.",
        "",
        _tabla(
            [
                "articulación",
                "ciclos",
                "desvío medio",
                "desvío máx",
                "marcado medio",
                "marcado máx",
            ],
            [
                [
                    nombre,
                    str(datos["ciclos"]),
                    f"{datos['desvio_medio']:.1f}°" if datos["desvio_medio"] else "—",
                    f"{datos['desvio_maximo']:.1f}°" if datos["desvio_maximo"] else "—",
                    f"{datos['cobertura_marcada_media']:.1%}",
                    f"{datos['cobertura_marcada_maxima']:.1%}",
                ]
                for nombre, datos in resumen["por_articulacion"].items()
            ],
        ),
        "",
        "## 4. Qué tan repetible es el corte",
        "",
        "Si el corte cayera siempre en el mismo punto del movimiento, el codo tendría que estar",
        "más o menos igual en todos los cortes. Estos son los valores que toma:",
        "",
        _tabla(
            ["fotograma del corte", *(str(corte) for corte in corte_codo["cortes"])],
            [["ángulo de codo", *(f"{valor:.0f}°" for valor in corte_codo["grados"])]],
        ),
        "",
        f"Van de **{corte_codo['minimo']:.0f}° a {corte_codo['maximo']:.0f}°** (mediana "
        f"{corte_codo['mediana']:.0f}°, desvío {corte_codo['desvio']:.0f}°). Es mucho para un",
        "instante que se supone el mismo. Eso explica el escalón de la curva media entre el 0 %",
        "y el 100 %, que son el mismo evento pero promediado sobre conjuntos de cortes distintos:",
        "el primer corte de cada tramo no es el 100 % de nadie y el último no es el 0 % de nadie.",
        "",
        "La dispersión puede venir de tres lados y con este material no se separan: que el",
        "detector no esté cortando siempre en el mismo punto, que el ángulo de codo esté muy",
        "ruidoso ahí, o que el nadador realmente llegue al recobro con el codo distinto cada vez.",
        "",
        "## 5. Qué tan lejos está esto de una medición",
        "",
        f"Con {codo['ciclos']} ciclos el desvío es informativo pero la muestra es chica: describe",
        "estos ciclos, no la brazada del nadador. Tres cosas acotan cuánto se puede leer acá:",
        "",
        f"- **El desvío es grande en términos de la señal.** ±{codo['desvio_medio']:.0f}° sobre",
        f"  una curva media que recorre {codo['excursion_de_la_media']:.0f}° es una banda ancha.",
        "  Parte de eso es variabilidad real entre ciclos y parte es ruido de los landmarks, y",
        "  con este material no se pueden separar.",
        "- **Casi un cuarto de las mediciones que entran está marcado.** No se descartan a",
        "  propósito, pero tampoco se puede afirmar que la curva no dependa de ellas.",
        "- **El corte no es tan repetible como debería.** El codo en el instante de corte va de",
        f"  {corte_codo['minimo']:.0f}° a {corte_codo['maximo']:.0f}°.",
        "",
        "Separar la variabilidad del nadador del error del método es exactamente lo que hace",
        "falta validar contra anotación manual, que sigue pendiente.",
        "",
        "---",
        "",
        f"Este informe describe {codo['ciclos']} ciclos de un video de desarrollo. Es material de",
        "construcción: sirve para verificar que el pipeline corre de punta a punta, no para",
        "sacar conclusiones sobre la técnica de nadie.",
        "",
    ]
    return "\n".join(partes)


if __name__ == "__main__":
    raise SystemExit(main())
