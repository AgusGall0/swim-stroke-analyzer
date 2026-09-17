"""Figuras del informe de caracterización de la señal.

Se generan a partir del Parquet ya persistido, nunca calculando una métrica al
paso solo para dibujarla.

La paleta y el estilo son los de :mod:`swimalyzer.viz.estilo`, comunes a todas
las figuras del proyecto.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from swimalyzer.pose import landmarks as lm
from swimalyzer.signal.caracterizacion import (
    SeriesDeLandmarks,
    analisis_de_residuos,
    espectro,
)
from swimalyzer.viz.estilo import (
    GRIS_SIN_DATO,
    RAMPA_AZUL,
    SUPERFICIE,
    TINTA,
    TINTA_SECUNDARIA,
    TINTA_TENUE,
    color_de_landmark,
    estilo,
    guardar,
    leyenda_de_lados,
)

#: Los seis segmentos, en orden próximo → distal.
SEGMENTOS: tuple[tuple[str, int, int], ...] = (
    ("hombro", lm.HOMBRO_IZQ, lm.HOMBRO_DER),
    ("codo", lm.CODO_IZQ, lm.CODO_DER),
    ("muñeca", lm.MUNECA_IZQ, lm.MUNECA_DER),
    ("cadera", lm.CADERA_IZQ, lm.CADERA_DER),
    ("rodilla", lm.RODILLA_IZQ, lm.RODILLA_DER),
    ("tobillo", lm.TOBILLO_IZQ, lm.TOBILLO_DER),
)

ORDEN_DE_LANDMARKS: tuple[int, ...] = tuple(
    landmark for _, izq, der in SEGMENTOS for landmark in (izq, der)
)


def figura_visibility(series: SeriesDeLandmarks, destino: Path) -> Path:
    """Distribución de visibility por landmark, con los umbrales candidatos."""
    estilo()
    figura, ejes = plt.subplots(figsize=(8, 5.2))
    detectado = series.detectado
    datos = [series.visibility[detectado, landmark] for landmark in ORDEN_DE_LANDMARKS]
    posiciones = np.arange(len(ORDEN_DE_LANDMARKS))

    caja = ejes.boxplot(
        datos,
        positions=posiciones,
        vert=False,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": SUPERFICIE, "lw": 1.6},
        whiskerprops={"color": TINTA_SECUNDARIA, "lw": 1.0},
        capprops={"color": TINTA_SECUNDARIA, "lw": 1.0},
    )
    for parche, landmark in zip(caja["boxes"], ORDEN_DE_LANDMARKS, strict=True):
        parche.set_facecolor(color_de_landmark(landmark))
        parche.set_edgecolor(SUPERFICIE)
        parche.set_linewidth(2)

    for umbral in (0.3, 0.5, 0.7, 0.9):
        ejes.axvline(umbral, color=TINTA_TENUE, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ejes.text(
            umbral,
            1.01,
            f"{umbral:g}",
            transform=ejes.get_xaxis_transform(),
            color=TINTA_SECUNDARIA,
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ejes.set_yticks(posiciones, [lm.nombre(landmark) for landmark in ORDEN_DE_LANDMARKS])
    ejes.invert_yaxis()
    ejes.set_xlim(0, 1)
    ejes.set_xlabel("visibility")
    ejes.set_title(
        f"Distribución de visibility por landmark\n"
        f"{int(detectado.sum())} fotogramas con detección de {series.cantidad_fotogramas}",
        loc="left",
        pad=16,
    )
    ejes.grid(axis="y", visible=False)
    leyenda_de_lados(figura)
    return guardar(figura, destino)


def figura_descartes(series: SeriesDeLandmarks, destino: Path) -> Path:
    """Porcentaje de fotogramas descartados en función del umbral de visibility."""
    estilo()
    figura, ejes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    umbrales = np.linspace(0, 1, 101)
    marcados = (0.3, 0.5, 0.7, 0.9)
    total = series.cantidad_fotogramas

    for eje, (nombre, izq, der) in zip(ejes.ravel(), SEGMENTOS, strict=True):
        for landmark in (izq, der):
            visibility = series.visibility[:, landmark]
            with np.errstate(invalid="ignore"):
                descartes = [
                    1 - np.sum(~np.isnan(visibility) & (visibility >= u)) / total for u in umbrales
                ]
            eje.plot(umbrales, np.array(descartes) * 100, color=color_de_landmark(landmark))
            eje.plot(
                marcados,
                [
                    (1 - np.sum(~np.isnan(visibility) & (visibility >= u)) / total) * 100
                    for u in marcados
                ],
                "o",
                ms=4.5,
                color=color_de_landmark(landmark),
                mec=SUPERFICIE,
                mew=1.2,
            )
        eje.set_title(nombre, loc="left")
        eje.set_ylim(0, 100)
        eje.set_xlim(0, 1)

    for eje in ejes[-1]:
        eje.set_xlabel("umbral de visibility")
    for eje in ejes[:, 0]:
        eje.set_ylabel("% de fotogramas descartados")
    figura.suptitle(
        "Cuánta serie temporal cuesta cada umbral\n"
        "sobre los 1055 fotogramas del video, contando los que no tuvieron detección",
        x=0.09,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TINTA,
    )
    leyenda_de_lados(figura)
    figura.tight_layout(rect=(0, 0.03, 1, 0.94))
    return guardar(figura, destino)


def figura_disponibilidad(series: SeriesDeLandmarks, destino: Path) -> Path:
    """Mapa fotograma × landmark: dónde hay dato y con qué visibility."""
    estilo()
    figura, ejes = plt.subplots(figsize=(11, 4))
    matriz = series.visibility[:, list(ORDEN_DE_LANDMARKS)].T
    mapa = RAMPA_AZUL.copy()
    mapa.set_bad(GRIS_SIN_DATO)
    imagen = ejes.imshow(
        np.ma.masked_invalid(matriz),
        aspect="auto",
        cmap=mapa,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(0, series.duracion_s, len(ORDEN_DE_LANDMARKS), 0),
    )
    ejes.set_yticks(
        np.arange(len(ORDEN_DE_LANDMARKS)) + 0.5,
        [lm.nombre(landmark) for landmark in ORDEN_DE_LANDMARKS],
    )
    ejes.set_xlabel("tiempo (s)")
    ejes.grid(visible=False)
    barra = figura.colorbar(imagen, ax=ejes, pad=0.01)
    barra.set_label("visibility", color=TINTA_SECUNDARIA)
    barra.outline.set_edgecolor(TINTA_TENUE)
    ejes.set_title(
        "Disponibilidad del dato a lo largo del video\n"
        "en gris, los fotogramas sin ninguna detección",
        loc="left",
    )
    return guardar(figura, destino)


def figura_espectros(series: SeriesDeLandmarks, tramo: tuple[int, int], destino: Path) -> Path:
    """Contenido frecuencial de las trayectorias en el tramo continuo más largo."""
    estilo()
    inicio, fin = tramo
    figura, ejes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
    for eje, (nombre, izq, der) in zip(ejes.ravel(), SEGMENTOS, strict=True):
        for landmark in (izq, der):
            for trazo, coordenada in (("-", series.x), ((0, (3, 2)), series.y)):
                frecuencias, potencia = espectro(coordenada[inicio:fin, landmark], series.fps)
                eje.semilogy(
                    frecuencias, potencia, color=color_de_landmark(landmark), ls=trazo, lw=1.3
                )
        eje.set_title(nombre, loc="left")
        eje.set_xlim(0, series.fps / 2)
    for eje in ejes[-1]:
        eje.set_xlabel("frecuencia (Hz)")
    for eje in ejes[:, 0]:
        eje.set_ylabel("potencia (px²/Hz)")
    figura.suptitle(
        f"Dónde está la señal y dónde empieza el ruido\n"
        f"tramo continuo de {(fin - inicio) / series.fps:.1f} s "
        f"(fotogramas {inicio} a {fin - 1}); línea llena eje x, punteada eje y",
        x=0.07,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TINTA,
    )
    leyenda_de_lados(figura)
    figura.tight_layout(rect=(0, 0.03, 1, 0.93))
    return guardar(figura, destino)


def figura_residuos(
    series: SeriesDeLandmarks, tramo: tuple[int, int], destino: Path, orden: int = 2
) -> Path:
    """Análisis de residuos de Winter: residuo RMS contra frecuencia de corte."""
    estilo()
    inicio, fin = tramo
    figura, ejes = plt.subplots(2, 3, figsize=(10, 5.8), sharex=True)
    for eje, (nombre, izq, der) in zip(ejes.ravel(), SEGMENTOS, strict=True):
        for landmark in (izq, der):
            residuos = analisis_de_residuos(series.x[inicio:fin, landmark], series.fps, orden=orden)
            color = color_de_landmark(landmark)
            eje.plot(residuos.cortes_hz, residuos.residuo_rms_px, color=color)
            eje.axhline(residuos.ruido_estimado_px, color=color, lw=0.9, ls=(0, (2, 2)))
            if np.isfinite(residuos.corte_optimo_hz):
                eje.plot(
                    [residuos.corte_optimo_hz],
                    [residuos.ruido_estimado_px],
                    "o",
                    ms=5,
                    color=color,
                    mec=SUPERFICIE,
                    mew=1.2,
                )
                eje.annotate(
                    f"{residuos.corte_optimo_hz:.1f} Hz",
                    (residuos.corte_optimo_hz, residuos.ruido_estimado_px),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=7,
                    color=color,
                )
        eje.set_title(nombre, loc="left")
    for eje in ejes[-1]:
        eje.set_xlabel("frecuencia de corte (Hz)")
    for eje in ejes[:, 0]:
        eje.set_ylabel("residuo RMS (px)")
    figura.suptitle(
        f"Análisis de residuos, coordenada x, Butterworth de orden {orden}\n"
        "la horizontal punteada es el ruido estimado; el punto, "
        "el corte donde el residuo lo iguala",
        x=0.07,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TINTA,
    )
    leyenda_de_lados(figura)
    figura.tight_layout(rect=(0, 0.03, 1, 0.93))
    return guardar(figura, destino)


def figura_intercambios(
    series: SeriesDeLandmarks,
    candidatos: dict[str, np.ndarray],
    tramo: tuple[int, int],
    destino: Path,
) -> Path:
    """Trayectorias horizontales de pares homólogos con los candidatos a intercambio."""
    estilo()
    inicio, fin = tramo
    pares = [seg for seg in SEGMENTOS if seg[0] in candidatos]
    figura, ejes = plt.subplots(len(pares), 1, figsize=(10, 2.3 * len(pares)), sharex=True)
    tiempo = np.arange(inicio, fin) / series.fps
    for eje, (nombre, izq, der) in zip(np.atleast_1d(ejes), pares, strict=True):
        for landmark in (izq, der):
            eje.plot(tiempo, series.x[inicio:fin, landmark], color=color_de_landmark(landmark))
        marcados = candidatos[nombre]
        marcados = marcados[(marcados >= inicio) & (marcados < fin)]
        for fotograma in marcados:
            eje.axvline(fotograma / series.fps, color=TINTA_TENUE, lw=1.0, zorder=0)
        eje.set_title(f"{nombre} — {len(marcados)} candidatos en este tramo", loc="left")
        eje.set_ylabel("x (px)")
    np.atleast_1d(ejes)[-1].set_xlabel("tiempo (s)")
    extra = [Line2D([], [], color=TINTA_TENUE, lw=1.0, label="candidato a intercambio")]
    figura.suptitle(
        "Trayectoria horizontal de los pares homólogos",
        x=0.07,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TINTA,
    )
    leyenda_de_lados(figura, extra)
    figura.tight_layout(rect=(0, 0.04, 1, 0.96))
    return guardar(figura, destino)
