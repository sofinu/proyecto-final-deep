"""Carga y verificación de los archivos originales de DrivenData."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def _ruta(nombre: str, directorio: Path | None) -> Path:
    ruta = Path(directorio or config.DIR_DATOS_RAW) / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descarga los datos de "
            "https://www.drivendata.org/competitions/57/nepal-earthquake/data/ "
            "y ubícalos en data/raw/ (ver data/README.md)."
        )
    return ruta


def cargar_train(directorio: Path | None = None) -> pd.DataFrame:
    """train_values + train_labels unidos 1 a 1 por building_id (260,601 filas)."""
    valores = pd.read_csv(_ruta(config.ARCHIVO_TRAIN_VALUES, directorio))
    etiquetas = pd.read_csv(_ruta(config.ARCHIVO_TRAIN_LABELS, directorio))
    if list(valores.columns) != config.COLUMNAS_ENTRADA:
        raise ValueError("train_values.csv no tiene las 39 columnas esperadas en el orden original.")
    # validate="one_to_one" lanza un error si algún building_id está repetido
    return valores.merge(etiquetas, on=config.ID, how="inner", validate="one_to_one")


def cargar_test(directorio: Path | None = None) -> pd.DataFrame:
    """test_values.csv (86,868 filas, sin etiqueta)."""
    return pd.read_csv(_ruta(config.ARCHIVO_TEST_VALUES, directorio))


def cargar_submission_format(directorio: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(_ruta(config.ARCHIVO_SUBMISSION_FORMAT, directorio))


def cargar_metadata(directorio: Path | None = None) -> dict:
    """Metadata del preprocesamiento original (orden de las 99 features, categorías, etc.)."""
    ruta = Path(directorio or config.DIR_DATOS_PROCESADOS) / config.ARCHIVO_METADATA
    return json.loads(ruta.read_text(encoding="utf-8"))
