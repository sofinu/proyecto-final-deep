"""Pipeline de inferencia: filas crudas (columnas de test_values.csv) -> damage_grade ∈ {1, 2, 3}.

Equivale a `predecir_dano_estructural()` de la Sección 6.2 del notebook, pero carga el
preprocesador y el ensamble desde `artifacts/models/` en lugar de depender de variables
en memoria del notebook.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .model import EnsambleM5, cargar_modelo
from .preprocessing import PreprocesadorRichter


class EntradaInvalidaError(ValueError):
    """Los datos de entrada no tienen la estructura que espera el modelo."""


@lru_cache(maxsize=4)
def _cargar_artefactos_cache(dir_modelos: str):
    d = Path(dir_modelos)
    ruta_pre = d / config.ARCHIVO_PREPROCESADOR
    if not ruta_pre.exists():
        raise FileNotFoundError(
            f"No existe {ruta_pre}. Genera los artefactos con `python -m richter_predictor.train` (ver README)."
        )
    return PreprocesadorRichter.cargar(ruta_pre), cargar_modelo(d / config.ARCHIVO_MODELO_M5)


def cargar_artefactos(dir_modelos: Path | None = None) -> tuple[PreprocesadorRichter, EnsambleM5]:
    return _cargar_artefactos_cache(str(Path(dir_modelos or config.DIR_MODELOS).resolve()))


def validar_entrada(df: pd.DataFrame, preprocesador: PreprocesadorRichter) -> pd.DataFrame:
    """Comprueba columnas, nulos, tipos y dominios. Devuelve una copia con tipos normalizados."""
    if not isinstance(df, pd.DataFrame):
        raise EntradaInvalidaError(f"Se esperaba un pandas.DataFrame y se recibió {type(df).__name__}.")
    if len(df) == 0:
        raise EntradaInvalidaError("El DataFrame de entrada está vacío.")

    faltantes = [c for c in config.COLUMNAS_ENTRADA if c not in df.columns]
    if faltantes:
        raise EntradaInvalidaError(
            f"Faltan {len(faltantes)} columnas obligatorias: {faltantes[:10]}"
            f"{' ...' if len(faltantes) > 10 else ''}. Se esperan las 39 columnas de test_values.csv."
        )
    df = df[config.COLUMNAS_ENTRADA].copy()

    con_nulos = df.columns[df.isna().any()].tolist()
    if con_nulos:
        raise EntradaInvalidaError(f"Hay valores nulos en: {con_nulos}. El modelo no imputa nulos.")
    if df[config.ID].duplicated().any():
        raise EntradaInvalidaError("Hay building_id repetidos en la entrada.")

    for c in [config.ID] + config.COLUMNAS_ENTERAS:
        convertida = pd.to_numeric(df[c], errors="coerce")
        if convertida.isna().any() or not np.all(np.mod(convertida, 1) == 0):
            ejemplo = df.loc[convertida.isna() | (np.mod(convertida.fillna(0), 1) != 0), c].iloc[0]
            raise EntradaInvalidaError(f"La columna '{c}' debe contener enteros; se encontró {ejemplo!r}.")
        df[c] = convertida.astype(np.int64)

    for c in config.BINARIAS:
        if not df[c].isin([0, 1]).all():
            raise EntradaInvalidaError(f"La columna binaria '{c}' solo admite 0 o 1.")
    for c in config.NUMERICAS:
        if (df[c] < 0).any():
            raise EntradaInvalidaError(f"La columna '{c}' tiene valores negativos.")

    # Categorías de baja cardinalidad: una categoría nueva quedaría codificada como "todo ceros",
    # una combinación que el modelo nunca vio. Se rechaza en lugar de predecir en silencio.
    for c, cats in preprocesador.categorias.items():
        nuevas = sorted(set(df[c].astype(str)) - set(cats))
        if nuevas:
            raise EntradaInvalidaError(
                f"La columna '{c}' tiene categorías que no existen en entrenamiento: {nuevas[:5]}. "
                f"Valores admitidos: {cats}."
            )
    # geo_level_2_id / geo_level_3_id no vistos sí se admiten: reciben la media global (igual que en el notebook).
    return df


def predecir_dano_estructural(df_crudo: pd.DataFrame, modelo: EnsambleM5 | None = None,
                              preprocesador: PreprocesadorRichter | None = None,
                              devolver_probabilidades: bool = False) -> pd.DataFrame:
    """Predice el nivel de daño de cada edificación.

    Parámetros
    ----------
    df_crudo : DataFrame con las 39 columnas de `test_values.csv` (building_id + 38 predictoras).
    modelo, preprocesador : opcionales; por defecto se cargan de `artifacts/models/`.
    devolver_probabilidades : si es True agrega las columnas `p_1`, `p_2`, `p_3`.

    Devuelve
    --------
    DataFrame con `building_id` y `damage_grade` (1, 2 o 3), en el mismo orden de la entrada.
    """
    if modelo is None or preprocesador is None:
        pre_def, mod_def = cargar_artefactos()
        preprocesador = preprocesador or pre_def
        modelo = modelo or mod_def

    df = validar_entrada(df_crudo, preprocesador)
    X = preprocesador.transform(df)
    if X.shape[1] != modelo.n_features_in_:
        raise EntradaInvalidaError(f"El preprocesamiento produjo {X.shape[1]} features y el modelo espera "
                                   f"{modelo.n_features_in_}.")
    # float32: el mismo tipo con el que se entrenó el ensamble en el notebook
    proba = modelo.predict_proba(X.to_numpy(np.float32))
    salida = pd.DataFrame({config.ID: df[config.ID].to_numpy(),
                           config.OBJETIVO: modelo.classes_[proba.argmax(1)].astype(np.int64)})
    if devolver_probabilidades:
        for k, c in enumerate(modelo.classes_):
            salida[f"p_{c}"] = proba[:, k]
    return salida
