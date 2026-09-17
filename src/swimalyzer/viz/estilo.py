"""Paleta y utilidades comunes a todas las figuras.

Color: el azul es el lado izquierdo del nadador y el naranja el derecho, en
todas las figuras. Son los dos primeros lugares de la paleta categórica de
referencia, que separan bien también para daltonismo; el eje x o y de una misma
trayectoria se distingue por tipo de línea, no por color.

El ámbar queda reservado para las marcas de calidad: un ángulo o una muestra
marcada no es de otro lado ni otra magnitud, así que no puede robarle un color
a la codificación de lado.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

IZQUIERDA = "#2a78d6"
DERECHA = "#eb6834"
MARCADO = "#d4a017"
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#a8a69c"
GRIS_SIN_DATO = "#e4e2dc"

#: Rampa secuencial de un solo tono (azul, claro → oscuro) para magnitudes.
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul_swimalyzer",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def estilo() -> None:
    """Aplica el estilo común de las figuras del proyecto."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": TINTA_TENUE,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.titlecolor": TINTA,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e8e6e0",
            "grid.linewidth": 0.8,
            "xtick.color": TINTA_SECUNDARIA,
            "ytick.color": TINTA_SECUNDARIA,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "lines.linewidth": 1.6,
        }
    )


def color_de_landmark(landmark_id: int) -> str:
    """Azul para los landmarks del lado izquierdo, naranja para los del derecho."""
    return IZQUIERDA if landmark_id % 2 == 1 else DERECHA


def guardar(figura: plt.Figure, destino: Path) -> Path:
    """Escribe la figura y cierra la ventana; devuelve la ruta."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return destino


def leyenda_de_lados(figura: plt.Figure, extra: list[Line2D] | None = None) -> None:
    """Leyenda al pie con la codificación de lado, más lo que haga falta agregar."""
    manijas = [
        Line2D([], [], color=IZQUIERDA, lw=2.4, label="izquierdo (lado cercano)"),
        Line2D([], [], color=DERECHA, lw=2.4, label="derecho (lado lejano)"),
    ]
    figura.legend(
        handles=manijas + (extra or []),
        loc="lower center",
        ncols=len(manijas) + len(extra or []),
        bbox_to_anchor=(0.5, -0.02),
    )
