"""Metadata de corrida: lo que hace falta para reproducir un resultado.

Sin fps, resolución, hash del modelo, versión del código y parámetros usados,
un Parquet de landmarks es un archivo de números sin procedencia. Todo eso se
escribe junto a los datos, en un JSON legible.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swimalyzer import __version__
from swimalyzer.config import Configuracion
from swimalyzer.io.video import InfoVideo

TAMANO_BLOQUE_BYTES = 1024 * 1024
TIEMPO_ESPERA_GIT_S = 5


def sha256_de_archivo(ruta: str | Path) -> str:
    """SHA-256 de un archivo, leído por bloques."""
    hash_ = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        while bloque := archivo.read(TAMANO_BLOQUE_BYTES):
            hash_.update(bloque)
    return hash_.hexdigest()


def _git(*argumentos: str) -> str | None:
    """Corre git en el repo del paquete; devuelve ``None`` si no se puede."""
    try:
        resultado = subprocess.run(
            ["git", *argumentos],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=TIEMPO_ESPERA_GIT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if resultado.returncode != 0:
        return None
    return resultado.stdout.strip()


def _version_de(paquete: str) -> str | None:
    try:
        modulo = __import__(paquete)
    except ImportError:
        return None
    return getattr(modulo, "__version__", None)


def _entorno() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "mediapipe": _version_de("mediapipe"),
        "cv2": _version_de("cv2"),
        "numpy": _version_de("numpy"),
        "pyarrow": _version_de("pyarrow"),
    }


def _codigo() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    estado = _git("status", "--porcelain")
    return {
        "version_swimalyzer": __version__,
        "commit_git": commit,
        # Si el árbol tenía cambios sin commitear, el commit no alcanza para
        # identificar el código que produjo esto.
        "arbol_git_limpio": None if estado is None else estado == "",
    }


def _modelo(configuracion: Configuracion) -> dict[str, Any]:
    ruta = configuracion.modelo.ruta
    return {
        "url": configuracion.modelo.url,
        "ruta": str(ruta),
        "sha256_esperado": configuracion.modelo.sha256,
        "sha256_archivo": sha256_de_archivo(ruta) if ruta.is_file() else None,
    }


def _video(info: InfoVideo, hashear_video: bool) -> dict[str, Any]:
    return {
        "archivo": str(info.ruta),
        "sha256": sha256_de_archivo(info.ruta) if hashear_video and info.ruta.is_file() else None,
        "fps": info.fps,
        "fps_declarado_por_el_contenedor": info.fps_declarado,
        "fps_forzado_desde_la_linea_de_comandos": info.fps_fue_forzado,
        "resolucion_original": [info.ancho_original, info.alto_original],
        # Resolución del fotograma que entró a la inferencia: es la que
        # convierte las coordenadas normalizadas a píxeles.
        "resolucion_inferencia": [info.ancho, info.alto],
        "recorte": None if info.recorte is None else info.recorte.model_dump(),
        "fotogramas_declarados_por_el_contenedor": info.fotogramas_declarados,
    }


def encabezado_de_corrida(etapa: str) -> dict[str, Any]:
    """Lo que toda corrida registra, sea de la etapa que sea."""
    return {
        "esquema_metadata": 1,
        "etapa": etapa,
        "fecha_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "codigo": _codigo(),
        "entorno": _entorno(),
    }


def construir_metadata(
    *,
    info: InfoVideo,
    configuracion: Configuracion,
    resultado: dict[str, Any],
    hashear_video: bool = True,
) -> dict[str, Any]:
    """Arma el diccionario de metadata de una corrida de extracción."""
    return {
        **encabezado_de_corrida("extraccion"),
        "video": _video(info, hashear_video),
        "modelo": _modelo(configuracion),
        # La configuración completa, no solo la sección de detección: así se
        # ve qué decisiones abiertas seguían en null cuando se corrió esto.
        "configuracion": json.loads(configuracion.model_dump_json()),
        "resultado": resultado,
    }


def escribir_metadata(metadata: dict[str, Any], destino: str | Path) -> Path:
    """Escribe la metadata como JSON legible."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destino
