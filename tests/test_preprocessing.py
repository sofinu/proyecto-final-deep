import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from richter_predictor import config
from richter_predictor.preprocessing import (PreprocesadorRichter, te_aplicar, te_estadisticos, te_oof)

from conftest import requiere_datos


@pytest.fixture
def ajustado(filas_sinteticas):
    y = pd.Series(np.random.default_rng(1).integers(1, 4, len(filas_sinteticas)))
    pre = PreprocesadorRichter()
    X_oof = pre.fit_transform_oof(filas_sinteticas, y)
    return pre, X_oof, y


# ------------------------------------------------------------ estructura
def test_dimensiones_y_orden_de_columnas(ajustado, filas_sinteticas, metadata):
    pre, X_oof, _ = ajustado
    X = pre.transform(filas_sinteticas)
    assert X.shape == (len(filas_sinteticas), 99)
    assert X_oof.shape == X.shape
    assert list(X.columns) == metadata["feature_cols"]
    assert pre.feature_cols == metadata["feature_cols"]


def test_sin_nulos(ajustado, filas_sinteticas):
    pre, X_oof, _ = ajustado
    assert not pre.transform(filas_sinteticas).isna().any().any()
    assert not X_oof.isna().any().any()


def test_one_hot_un_uno_por_variable(ajustado, filas_sinteticas):
    pre, _, _ = ajustado
    X = pre.transform(filas_sinteticas)
    for c, cats in pre.categorias.items():
        assert (X[[f"{c}_{k}" for k in cats]].sum(axis=1) == 1).all()


def test_binarias_sin_transformar(ajustado, filas_sinteticas):
    pre, _, _ = ajustado
    X = pre.transform(filas_sinteticas)
    assert (X[config.BINARIAS].to_numpy() == filas_sinteticas[config.BINARIAS].to_numpy()).all()


# ------------------------------------------------------- valores especiales
def test_age_995_bandera_e_imputacion(ajustado, filas_sinteticas):
    pre, _, _ = ajustado
    X = pre.transform(filas_sinteticas)
    es_995 = filas_sinteticas["age"].to_numpy() == 995
    assert es_995.any()
    assert (X["age_valor_especial_995"].to_numpy() == es_995).all()
    mu, sd = pre.scaler["age"]
    esperado = (pre.mediana_age - mu) / sd
    assert np.allclose(X.loc[es_995, "age_s"], esperado)
    # la mediana se calcula sin el código 995
    edades = filas_sinteticas["age"]
    assert pre.mediana_age == edades[edades != 995].median()


def test_categoria_geo_no_vista_recibe_la_media_global(ajustado, filas_sinteticas):
    pre, _, _ = ajustado
    fila = filas_sinteticas.iloc[[0]].copy()
    fila["geo_level_3_id"] = 999_999
    X = pre.transform(fila)
    mu, sd = pre.te_escala_oof["geo_level_3_id"]
    assert np.isclose(X["geo_level_3_id_target_enc_s"].iloc[0], (pre.prior - mu) / sd)


# ---------------------------------------------------- target encoding OOF
def test_te_oof_no_usa_la_etiqueta_propia():
    """Cada fila se codifica solo con estadísticos de los otros pliegues."""
    rng = np.random.default_rng(7)
    geo = pd.Series(rng.integers(0, 15, 300))
    y = pd.Series(rng.integers(1, 4, 300).astype(float))
    prior = y.mean()
    oof = te_oof(geo, y, prior)

    skf = StratifiedKFold(n_splits=config.N_FOLDS_OOF, shuffle=True, random_state=config.SEMILLA)
    for idx_fit, idx_enc in skf.split(geo, y):
        st = te_estadisticos(geo.iloc[idx_fit], y.iloc[idx_fit])
        esperado = te_aplicar(geo.iloc[idx_enc], st, prior).to_numpy()
        assert np.allclose(oof[idx_enc], esperado)
        # ninguna fila codificada aparece en los datos con que se calcularon sus estadísticos
        assert not set(idx_fit) & set(idx_enc)


def test_te_oof_categoria_de_un_solo_edificio():
    """Con OOF, un edificio único en su zona recibe la media global; el encoding 'congelado' filtra su etiqueta."""
    geo = pd.Series([1] * 50 + [2] * 50 + [99])
    y = pd.Series([1.0, 2.0, 3.0] * 33 + [2.0, 3.0])
    prior = y.mean()
    oof = te_oof(geo, y, prior)
    assert np.isclose(oof[-1], prior)

    congelado = te_aplicar(geo, te_estadisticos(geo, y), prior).to_numpy()
    m = config.SMOOTHING_M
    assert np.isclose(congelado[-1], (y.iloc[-1] + m * prior) / (1 + m))


def test_guardar_y_cargar_reproduce_la_transformacion(ajustado, filas_sinteticas, tmp_path):
    pre, _, _ = ajustado
    ruta = pre.guardar(tmp_path / "pre.json")
    pre2 = PreprocesadorRichter.cargar(ruta)
    a = pre.transform(filas_sinteticas).to_numpy()
    b = pre2.transform(filas_sinteticas).to_numpy()
    assert np.array_equal(a, b)


def test_transform_sin_ajustar_falla(filas_sinteticas):
    with pytest.raises(RuntimeError):
        PreprocesadorRichter().transform(filas_sinteticas)


# ------------------------------------------- trazabilidad con datos reales
@requiere_datos
def test_reproduce_split_y_parquet_original():
    """El preprocesamiento de src/ reproduce el parquet histórico (encoding congelado) y el split."""
    from richter_predictor.data_loader import cargar_train
    from richter_predictor.preprocessing import mascara_validacion

    ruta_parquet = config.DIR_DATOS_PROCESADOS / "terremoto_train_procesado.parquet"
    if not ruta_parquet.exists():
        pytest.skip("falta data/processed/terremoto_train_procesado.parquet")
    datos = cargar_train()
    proc = pd.read_parquet(ruta_parquet)
    es_val = mascara_validacion(datos)
    assert (np.where(es_val, "val", "train") == proc.split.to_numpy()).all()
    assert es_val.sum() == 52_121

    pre = PreprocesadorRichter().fit(datos.loc[~es_val, config.COLUMNAS_ENTRADA], datos.loc[~es_val, config.OBJETIVO])
    X_val = pre.transform(datos.loc[es_val, config.COLUMNAS_ENTRADA], escala_te="congelado")
    dif = np.abs(X_val.to_numpy() - proc.loc[es_val, pre.feature_cols].to_numpy(float)).max()
    assert dif < 1e-9
