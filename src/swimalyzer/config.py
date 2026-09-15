"""Carga y validación de ``config.yaml``.

Todos los parámetros son obligatorios: si falta una clave, o sobra una
desconocida, la carga falla con un mensaje que nombra el parámetro. Los
parámetros que corresponden a decisiones abiertas admiten ``null`` al cargar,
pero la etapa que los use tiene que pedirlos con :meth:`Configuracion.exigir`,
que falla si siguen sin definir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Probabilidad = Annotated[float, Field(ge=0, le=1)]


class ErrorDeConfiguracion(Exception):
    """La configuración no se pudo cargar o le falta un valor que se necesita."""


class _Seccion(BaseModel):
    # extra="forbid": una clave mal escrita es un error, no un parámetro ignorado.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Modelo(_Seccion):
    url: Annotated[str, Field(pattern=r"^https://")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    ruta: Path


class Deteccion(_Seccion):
    num_poses: Annotated[int, Field(ge=1)]
    min_pose_detection_confidence: Probabilidad
    min_pose_presence_confidence: Probabilidad
    min_tracking_confidence: Probabilidad


class Recorte(_Seccion):
    x: Annotated[int, Field(ge=0)]
    y: Annotated[int, Field(ge=0)]
    ancho: Annotated[int, Field(gt=0)]
    alto: Annotated[int, Field(gt=0)]


class Video(_Seccion):
    recorte: Recorte | None


class Calidad(_Seccion):
    umbral_visibility: Probabilidad | None


class Filtrado(_Seccion):
    tipo: str | None
    orden: Annotated[int, Field(ge=1)] | None
    frecuencia_corte_hz: Annotated[float, Field(gt=0)] | None


class Segmentacion(_Seccion):
    senal: str | None
    distancia_minima_entre_picos_s: Annotated[float, Field(gt=0)] | None


class Lateralidad(_Seccion):
    metodo_correccion_intercambios: str | None


class Configuracion(_Seccion):
    modelo: Modelo
    deteccion: Deteccion
    video: Video
    calidad: Calidad
    filtrado: Filtrado
    segmentacion: Segmentacion
    lateralidad: Lateralidad

    def exigir(self, parametro: str) -> Any:
        """Devuelve el valor de ``parametro`` y falla si está en ``null``.

        ``parametro`` es la ruta con puntos, por ejemplo
        ``"filtrado.frecuencia_corte_hz"``. Lanza ``KeyError`` si la ruta no
        existe (error de programación) y :class:`ErrorDeConfiguracion` si el
        valor no está definido.
        """
        valor: Any = self
        for parte in parametro.split("."):
            if not isinstance(valor, BaseModel) or parte not in type(valor).model_fields:
                raise KeyError(f"'{parametro}' no es un parámetro de la configuración")
            valor = getattr(valor, parte)
        if valor is None:
            raise ErrorDeConfiguracion(
                f"El parámetro '{parametro}' está en null y esta etapa lo necesita. "
                "Si es una de las decisiones abiertas de CLAUDE.md, el valor se "
                "justifica con evidencia antes de completarlo en config.yaml."
            )
        return valor


def cargar_configuracion(archivo: str | Path) -> Configuracion:
    """Lee y valida un archivo de configuración YAML.

    Las rutas relativas del archivo (``modelo.ruta``) se resuelven respecto de
    la carpeta que contiene la configuración, no del directorio de trabajo.
    """
    archivo = Path(archivo)
    try:
        texto = archivo.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ErrorDeConfiguracion(f"No existe el archivo de configuración: {archivo}") from None
    except OSError as error:
        raise ErrorDeConfiguracion(f"No se pudo leer {archivo}: {error}") from error

    try:
        datos = yaml.safe_load(texto)
    except yaml.YAMLError as error:
        raise ErrorDeConfiguracion(f"{archivo} no es YAML válido:\n{error}") from error

    if not isinstance(datos, dict):
        raise ErrorDeConfiguracion(
            f"{archivo} tiene que contener un mapeo de secciones (modelo, deteccion, ...)"
        )

    try:
        configuracion = Configuracion.model_validate(datos)
    except ValidationError as error:
        raise ErrorDeConfiguracion(_describir_errores(error, archivo)) from None

    ruta_modelo = configuracion.modelo.ruta
    if not ruta_modelo.is_absolute():
        modelo = configuracion.modelo.model_copy(
            update={"ruta": (archivo.parent / ruta_modelo).resolve()}
        )
        configuracion = configuracion.model_copy(update={"modelo": modelo})
    return configuracion


def _describir_errores(error: ValidationError, archivo: Path) -> str:
    lineas = [f"Configuración inválida en {archivo}:"]
    for detalle in error.errors():
        parametro = ".".join(str(parte) for parte in detalle["loc"])
        if detalle["type"] == "missing":
            lineas.append(f"  - falta el parámetro obligatorio '{parametro}'")
        elif detalle["type"] == "extra_forbidden":
            lineas.append(f"  - parámetro desconocido '{parametro}' (¿error de tipeo?)")
        else:
            lineas.append(
                f"  - '{parametro}': {detalle['msg']} (valor recibido: {detalle['input']!r})"
            )
    return "\n".join(lineas)
