"""Figuras del ciclo de brazada: dónde se cortó y qué curva media salió.

Se dibujan a partir de lo que la segmentación ya persistió. La curva media no
esconde de qué está hecha: debajo va la cobertura, que dice en cada fase qué
fracción de las mediciones promediadas estaba marcada.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from swimalyzer.metrics.ciclos import Ciclo, CurvaMedia, SenalDeSegmentacion
from swimalyzer.signal.caracterizacion import tramos_continuos
from swimalyzer.viz.estilo import (
    IZQUIERDA,
    MARCADO,
    TINTA_SECUNDARIA,
    TINTA_TENUE,
    estilo,
    guardar,
)

#: Por debajo de esto la muestra es demasiado chica para leer el desvío como
#: una dispersión estable, y la figura lo dice en vez de dejarlo implícito.
CICLOS_PARA_MUESTRA_HOLGADA = 10


def _epigrafe_de_muestra(curva: CurvaMedia) -> str:
    """Qué dice la figura sobre el tamaño de su propia muestra."""
    # Numerados desde 0, igual que en el Parquet y la metadata: dos numeraciones
    # distintas para lo mismo es la forma más fácil de leer mal una figura.
    tramos = ", ".join(str(tramo) for tramo in curva.tramos)
    frase = (
        f"{curva.n} ciclos promediados, de {len(curva.tramos)} "
        f"tramo{'s' if len(curva.tramos) != 1 else ''} continuo"
        f"{'s' if len(curva.tramos) != 1 else ''} (nº {tramos})."
    )
    if curva.n < CICLOS_PARA_MUESTRA_HOLGADA:
        frase += (
            f" Con {curva.n} ciclos el desvío es informativo pero la muestra es chica: "
            "describe estos ciclos, no la brazada del nadador."
        )
    return frase


def figura_curva_media(
    curva: CurvaMedia,
    destino: Path,
    titulo: str | None = None,
) -> Path:
    """Curva media ± 1 desvío estándar del ángulo a lo largo del ciclo.

    El panel de arriba es la curva; el de abajo, la cobertura. Van juntos porque
    ningún ciclo se descartó por tener mediciones marcadas: con esta cantidad de
    ciclos descartar deja la muestra en nada y esconde el problema, así que la
    figura incluye todo y muestra de qué está hecha cada fase.
    """
    estilo()
    figura, (arriba, abajo) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        height_ratios=(3, 1),
        gridspec_kw={"hspace": 0.12},
    )

    # Los ciclos individuales, tenues: son la muestra de la que sale la media y
    # con siete curvas se pueden mirar una por una.
    for fila in range(curva.n):
        arriba.plot(curva.malla, curva.curvas[fila], color=IZQUIERDA, lw=0.8, alpha=0.3)

    if curva.n > 1:
        arriba.fill_between(
            curva.malla,
            curva.media - curva.desvio,
            curva.media + curva.desvio,
            color=IZQUIERDA,
            alpha=0.18,
            lw=0,
        )
    arriba.plot(curva.malla, curva.media, color=IZQUIERDA, lw=2.4)

    arriba.set_ylim(0, 185)
    arriba.set_yticks(np.arange(0, 181, 30))
    arriba.set_ylabel("ángulo (grados)")
    arriba.axhline(180, color=TINTA_TENUE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    arriba.text(
        0,
        181,
        "180° = extendido",
        fontsize=7,
        color=TINTA_SECUNDARIA,
        va="bottom",
    )

    abajo.fill_between(
        curva.malla,
        0,
        100 * curva.cobertura_marcada,
        color=MARCADO,
        alpha=0.45,
        lw=0,
    )
    abajo.plot(curva.malla, 100 * curva.cobertura_marcada, color=MARCADO, lw=1.4)
    abajo.set_ylim(0, 100)
    abajo.set_yticks((0, 50, 100))
    abajo.set_ylabel("marcado\n(% de ciclos)", fontsize=8)
    abajo.set_xlabel("porcentaje del ciclo de brazada")
    abajo.set_xlim(0, 100)
    abajo.set_xticks(np.arange(0, 101, 10))

    desvio_medio = float(np.nanmean(curva.desvio)) if curva.n > 1 else float("nan")
    arriba.set_title(
        titulo
        or (
            f"Ángulo de {curva.articulacion.replace('_', ' ')} a lo largo del ciclo de brazada\n"
            f"media ± 1 desvío estándar de {curva.n} ciclos; "
            f"desvío medio {desvio_medio:.1f}°"
        ),
        loc="left",
        pad=14,
    )
    figura.legend(
        handles=[
            Line2D([], [], color=IZQUIERDA, lw=2.4, label="curva media"),
            Patch(facecolor=IZQUIERDA, alpha=0.18, label="± 1 desvío estándar"),
            Line2D([], [], color=IZQUIERDA, lw=0.8, alpha=0.5, label="ciclo individual"),
            Patch(facecolor=MARCADO, alpha=0.45, label="mediciones marcadas"),
        ],
        loc="lower center",
        ncols=4,
        bbox_to_anchor=(0.5, -0.03),
    )
    figura.text(
        0.0,
        -0.075,
        _epigrafe_de_muestra(curva)
        + " Ningún ciclo se descartó por tener mediciones marcadas: el panel de"
        " abajo dice qué fracción lo está en cada fase.",
        fontsize=7.5,
        color=TINTA_SECUNDARIA,
        wrap=True,
    )
    figura.text(
        0.0,
        -0.115,
        "0 % y 100 % son el mismo evento —la mano en lo más alto del recobro— en dos"
        " repeticiones consecutivas. Ángulo proyectado en el plano de la imagen.",
        fontsize=7.5,
        color=TINTA_SECUNDARIA,
    )
    return guardar(figura, destino)


def figura_segmentacion(
    senal: np.ndarray,
    disponible: np.ndarray,
    ciclos: list[Ciclo],
    fps: float,
    descripcion: SenalDeSegmentacion,
    destino: Path,
    tramo: tuple[int, int] | None = None,
) -> Path:
    """La señal de corte con los cortes marcados, para ver dónde cayó cada ciclo.

    Es la figura que permite discutir la segmentación: sin ver dónde cortó, la
    curva media es un promedio de algo que hay que creer de palabra.
    """
    estilo()
    if tramo is None:
        tramos = tramos_continuos(disponible)
        tramo = max(tramos, key=lambda par: par[1] - par[0]) if tramos else (0, senal.size)
    inicio, fin = tramo
    tiempo = np.arange(inicio, fin) / fps
    valores = np.where(disponible[inicio:fin], senal[inicio:fin], np.nan)

    figura, ejes = plt.subplots(figsize=(11, 3.8))
    ejes.axhline(0, color=TINTA_TENUE, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ejes.plot(tiempo, valores, color=IZQUIERDA)

    del_tramo = [ciclo for ciclo in ciclos if inicio <= ciclo.inicio < fin]
    cortes = sorted({ciclo.inicio for ciclo in del_tramo} | {ciclo.fin for ciclo in del_tramo})
    for corte in cortes:
        ejes.axvline(corte / fps, color=MARCADO, lw=1.2, alpha=0.9)
    ejes.plot(
        [corte / fps for corte in cortes],
        [senal[corte] for corte in cortes],
        "v",
        ms=7,
        color=MARCADO,
        mec="none",
    )
    for ciclo in del_tramo:
        ejes.annotate(
            f"{ciclo.duracion_s(fps):.2f} s",
            xy=((ciclo.inicio + ciclo.fin) / 2 / fps, ejes.get_ylim()[1]),
            ha="center",
            va="top",
            fontsize=7.5,
            color=TINTA_SECUNDARIA,
        )

    ejes.set_xlim(tiempo[0], tiempo[-1])
    ejes.set_xlabel("tiempo (s)")
    ejes.set_ylabel("píxeles")
    ejes.set_title(
        f"Señal de corte: {descripcion.descripcion}\n"
        f"el máximo es {descripcion.evento}; {len(del_tramo)} ciclos en este tramo",
        loc="left",
        pad=12,
        fontsize=9.5,
    )
    figura.legend(
        handles=[
            Line2D([], [], color=IZQUIERDA, lw=2.4, label=descripcion.nombre),
            Line2D([], [], color=MARCADO, lw=1.6, label="corte de ciclo"),
        ],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.06),
    )
    figura.text(
        0.0,
        -0.14,
        "Por encima de 0 la muñeca está más arriba que el hombro en la imagen.",
        fontsize=7.5,
        color=TINTA_SECUNDARIA,
    )
    return guardar(figura, destino)
