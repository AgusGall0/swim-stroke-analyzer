"""Figuras de las series de ángulos articulares.

Se dibujan a partir del Parquet de ángulos ya persistido: acá no se calcula
ninguna métrica al paso. El ámbar señala lo marcado; el valor marcado se dibuja
igual que el resto, porque marcar no es descartar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from swimalyzer.metrics.angulos import MOTIVOS, SerieDeAngulo
from swimalyzer.signal.caracterizacion import tramos_continuos
from swimalyzer.viz.estilo import (
    IZQUIERDA,
    MARCADO,
    SUPERFICIE,
    TINTA,
    TINTA_SECUNDARIA,
    TINTA_TENUE,
    estilo,
    guardar,
)

#: Cómo se lee cada motivo en una figura.
ETIQUETAS_DE_MOTIVO: dict[str, str] = {
    "sin_filtrar": "sin filtrar",
    "interpolado": "interpolado",
    "intercambio_sospechado": "intercambio sospechado",
    "visibility_baja": "visibility baja",
}


def figura_serie(
    serie: SerieDeAngulo,
    fps: float,
    tramo: tuple[int, int],
    destino: Path,
    titulo: str | None = None,
) -> Path:
    """Serie temporal de un ángulo en un tramo, con sus marcas desglosadas.

    El panel de arriba es el ángulo; el de abajo dice, fotograma a fotograma,
    por qué motivo quedó marcado. Van juntos porque mirar la curva sin saber
    qué parte de ella es dudosa es la forma más fácil de creerle de más.
    """
    estilo()
    inicio, fin = tramo
    tiempo = np.arange(inicio, fin) / fps
    grados = serie.grados[inicio:fin]
    marcado = serie.marcado[inicio:fin]
    motivos = {motivo: serie.motivos[motivo][inicio:fin] for motivo in MOTIVOS}

    figura, (arriba, abajo) = plt.subplots(
        2,
        1,
        figsize=(10, 5.4),
        sharex=True,
        height_ratios=(3, 1),
        gridspec_kw={"hspace": 0.12},
    )

    for desde, hasta in tramos_continuos(marcado):
        arriba.axvspan(
            tiempo[desde],
            tiempo[min(hasta, tiempo.size - 1)],
            color=MARCADO,
            alpha=0.14,
            lw=0,
            zorder=0,
        )
    arriba.plot(tiempo, grados, color=IZQUIERDA)
    # Los huecos sin ángulo se ven como corte de la línea; los puntos sueltos
    # que quedan aislados entre NaN no se dibujarían, así que van con marcador.
    aislado = ~np.isnan(grados)
    aislado[1:-1] &= np.isnan(grados[:-2]) & np.isnan(grados[2:])
    arriba.plot(tiempo[aislado], grados[aislado], "o", ms=3, color=IZQUIERDA)

    arriba.set_ylim(0, 185)
    arriba.set_yticks(np.arange(0, 181, 30))
    arriba.set_ylabel("ángulo (grados)")
    arriba.axhline(180, color=TINTA_TENUE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    arriba.text(
        tiempo[0],
        181,
        "180° = extendido",
        fontsize=7,
        color=TINTA_SECUNDARIA,
        va="bottom",
    )

    filas = list(MOTIVOS)
    for numero, motivo in enumerate(filas):
        marcas = motivos[motivo]
        abajo.plot(
            tiempo[marcas],
            np.full(int(marcas.sum()), numero),
            marker="|",
            ls="none",
            ms=9,
            mew=1.4,
            color=MARCADO,
        )
    abajo.set_yticks(range(len(filas)), [ETIQUETAS_DE_MOTIVO[motivo] for motivo in filas])
    abajo.set_ylim(len(filas) - 0.5, -0.5)
    abajo.set_xlabel("tiempo (s)")
    abajo.grid(axis="y", visible=False)
    abajo.set_xlim(tiempo[0], tiempo[-1])

    con_dato = int((~np.isnan(grados)).sum())
    tasa = marcado.sum() / con_dato if con_dato else 0.0
    arriba.set_title(
        titulo
        or (
            f"Ángulo de {serie.articulacion.nombre.replace('_', ' ')} "
            f"({serie.articulacion.descripcion})\n"
            f"tramo continuo de {(fin - inicio) / fps:.1f} s (fotogramas {inicio} a {fin - 1}); "
            f"{con_dato} ángulos, {marcado.sum()} marcados ({tasa:.1%})"
        ),
        loc="left",
        pad=14,
    )
    figura.legend(
        handles=[
            Line2D([], [], color=IZQUIERDA, lw=2.4, label="ángulo medido (lado cercano)"),
            Patch(facecolor=MARCADO, alpha=0.3, label="fotograma marcado"),
        ],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.04),
    )
    figura.text(
        0.0,
        -0.085,
        "Ángulo proyectado en el plano de la imagen: con el cuerpo rotado, el valor medido "
        "es menor que el real.",
        fontsize=7.5,
        color=TINTA_SECUNDARIA,
    )
    return guardar(figura, destino)


def figura_rangos(series: dict[str, SerieDeAngulo], destino: Path) -> Path:
    """Distribución de cada ángulo sobre toda la corrida, marcados aparte.

    Sirve para lo que la serie temporal no muestra: si los valores que toma
    cada articulación caen dentro de lo anatómicamente posible.
    """
    estilo()
    figura, ejes = plt.subplots(figsize=(8, 0.95 * len(series) + 1.8))
    nombres = list(series)

    for numero, nombre in enumerate(nombres):
        serie = series[nombre]
        for desplazamiento, mascara, color in (
            (-0.16, serie.limpio, IZQUIERDA),
            (0.16, serie.marcado, MARCADO),
        ):
            valores = serie.grados[mascara]
            if valores.size == 0:
                continue
            ejes.boxplot(
                valores,
                positions=[numero + desplazamiento],
                vert=False,
                widths=0.26,
                showfliers=True,
                patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": SUPERFICIE, "lw": 2},
                medianprops={"color": SUPERFICIE, "lw": 1.6},
                whiskerprops={"color": TINTA_SECUNDARIA, "lw": 1.0},
                capprops={"color": TINTA_SECUNDARIA, "lw": 1.0},
                flierprops={
                    "marker": ".",
                    "ms": 3,
                    "mfc": color,
                    "mec": "none",
                    "alpha": 0.5,
                },
            )

    ejes.set_yticks(range(len(nombres)), [nombre.replace("_", " ") for nombre in nombres])
    ejes.set_ylim(len(nombres) - 0.5, -0.5)
    ejes.set_xlim(0, 185)
    ejes.set_xticks(np.arange(0, 181, 30))
    ejes.set_xlabel("ángulo (grados)")
    ejes.grid(axis="y", visible=False)
    ejes.axvline(180, color=TINTA_TENUE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ejes.set_title(
        "Qué valores toma cada ángulo en toda la corrida\n"
        "180° es el segmento extendido; no hay valores posibles por encima",
        loc="left",
        pad=14,
    )
    figura.legend(
        handles=[
            Patch(facecolor=IZQUIERDA, label="sin marcar"),
            Patch(facecolor=MARCADO, label="marcado"),
        ],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.02),
    )
    figura.text(
        0.0,
        -0.06,
        "Ángulo proyectado en el plano de la imagen.",
        fontsize=7.5,
        color=TINTA_SECUNDARIA,
    )
    figura.set_facecolor(SUPERFICIE)
    ejes.set_facecolor(SUPERFICIE)
    ejes.title.set_color(TINTA)
    return guardar(figura, destino)
