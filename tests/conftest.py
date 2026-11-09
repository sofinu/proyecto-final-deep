"""Fixtures compartidas.

Las pruebas se dividen en tres grupos:
- sin dependencias: usan datos sintéticos construidos con las categorías del metadata;
- `artefactos`: necesitan artifacts/models/ (versionado en el repositorio);
- `datos`: necesitan los CSV de DrivenData en data/raw/ (no versionados); se omiten si faltan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from richter_predictor import config
from richter_predictor.data_loader import cargar_metadata


@pytest.fixture(scope="session")
def metadata():
    return cargar_metadata()


def construir_filas_sinteticas(n: int, metadata: dict, semilla: int = 0) -> pd.DataFrame:
    """Filas crudas válidas con las 39 columnas de test_values.csv.

    Cada categoría de baja cardinalidad aparece al menos una vez (n >= 31).
    """
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({config.ID: np.arange(1, n + 1)})
    for c in config.COLUMNAS_ENTRADA[1:]:
        if c in metadata["categorias_por_columna"]:
            cats = metadata["categorias_por_columna"][c]
            valores = [cats[i % len(cats)] for i in range(n)]
            rng.shuffle(valores)
            df[c] = [int(v) for v in valores] if c == "geo_level_1_id" else valores
        elif c in config.BINARIAS:
            df[c] = rng.integers(0, 2, n)
        elif c == "geo_level_2_id":
            df[c] = rng.integers(0, 30, n)
        elif c == "geo_level_3_id":
            df[c] = rng.integers(0, 120, n)
        elif c == "age":
            df[c] = rng.choice([0, 5, 10, 15, 20, 40, 995], n)
        elif c == "count_floors_pre_eq":
            df[c] = rng.integers(1, 5, n)
        elif c == "area_percentage":
            df[c] = rng.integers(1, 30, n)
        elif c == "height_percentage":
            df[c] = rng.integers(2, 12, n)
        elif c == "count_families":
            df[c] = rng.integers(0, 3, n)
    return df[config.COLUMNAS_ENTRADA]


@pytest.fixture
def filas_sinteticas(metadata):
    return construir_filas_sinteticas(400, metadata)


def _datos_disponibles() -> bool:
    return all((config.DIR_DATOS_RAW / f).exists() for f in (config.ARCHIVO_TRAIN_VALUES, config.ARCHIVO_TRAIN_LABELS,
                                                             config.ARCHIVO_TEST_VALUES,
                                                             config.ARCHIVO_SUBMISSION_FORMAT))


def _artefactos_disponibles() -> bool:
    return all((config.DIR_MODELOS / f).exists() for f in (config.ARCHIVO_PREPROCESADOR, config.ARCHIVO_MODELO_M5))


requiere_datos = pytest.mark.skipif(not _datos_disponibles(), reason="faltan los CSV de DrivenData en data/raw/")
requiere_artefactos = pytest.mark.skipif(not _artefactos_disponibles(),
                                         reason="faltan los artefactos del modelo en artifacts/models/")
