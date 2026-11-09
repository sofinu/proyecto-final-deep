import numpy as np
import pandas as pd
import pytest

from richter_predictor import config
from richter_predictor.pipeline import EntradaInvalidaError, predecir_dano_estructural

from conftest import requiere_artefactos, requiere_datos

pytestmark = requiere_artefactos

RUTA_SUBMISSION = config.DIR_PREDICCIONES / "submission_M5.csv"


# -------------------------------------------------------- entradas válidas
def test_acepta_muestra_valida_y_formato_de_salida(filas_sinteticas):
    pred = predecir_dano_estructural(filas_sinteticas)
    assert list(pred.columns) == ["building_id", "damage_grade"]
    assert len(pred) == len(filas_sinteticas)
    assert (pred.building_id.to_numpy() == filas_sinteticas.building_id.to_numpy()).all()
    assert pred.damage_grade.isin([1, 2, 3]).all()
    assert pred.damage_grade.dtype == np.int64


def test_probabilidades_opcionales(filas_sinteticas):
    pred = predecir_dano_estructural(filas_sinteticas.head(20), devolver_probabilidades=True)
    proba = pred[["p_1", "p_2", "p_3"]].to_numpy()
    assert np.allclose(proba.sum(axis=1), 1)
    assert (pred.damage_grade.to_numpy() == proba.argmax(1) + 1).all()


def test_consistencia_por_lotes_y_fila_a_fila(filas_sinteticas):
    """Una fila se predice igual sola que dentro de un lote (no se ajusta nada con la entrada)."""
    lote = predecir_dano_estructural(filas_sinteticas.head(15))
    individual = pd.concat([predecir_dano_estructural(filas_sinteticas.iloc[[i]]) for i in range(15)])
    assert (lote.damage_grade.to_numpy() == individual.damage_grade.to_numpy()).all()


def test_columnas_extra_y_orden_distinto_se_admiten(filas_sinteticas):
    base = predecir_dano_estructural(filas_sinteticas.head(10))
    otra = filas_sinteticas.head(10)[config.COLUMNAS_ENTRADA[::-1]].assign(columna_extra=1)
    assert base.equals(predecir_dano_estructural(otra))


def test_geo_no_visto_se_admite(filas_sinteticas):
    df = filas_sinteticas.head(5).copy()
    df["geo_level_3_id"] = 10_000_000
    assert predecir_dano_estructural(df).damage_grade.isin([1, 2, 3]).all()


# ------------------------------------------------------ entradas inválidas
@pytest.mark.parametrize("modificar, mensaje", [
    (lambda df: df.drop(columns=["foundation_type"]), "Faltan"),
    (lambda df: df.assign(age=np.nan), "nulos"),
    (lambda df: df.assign(roof_type="zz"), "categorías que no existen"),
    (lambda df: df.assign(has_superstructure_timber=2), "solo admite 0 o 1"),
    (lambda df: df.assign(count_floors_pre_eq="dos"), "enteros"),
    (lambda df: df.assign(area_percentage=-3), "negativos"),
    (lambda df: df.assign(building_id=1), "repetidos"),
    (lambda df: df.iloc[0:0], "vacío"),
])
def test_rechaza_entradas_invalidas(filas_sinteticas, modificar, mensaje):
    with pytest.raises(EntradaInvalidaError, match=mensaje):
        predecir_dano_estructural(modificar(filas_sinteticas.head(10).copy()))


def test_rechaza_tipos_que_no_son_dataframe():
    with pytest.raises(EntradaInvalidaError):
        predecir_dano_estructural([[1, 2, 3]])


# --------------------------------------------- datos reales y archivo de envío
@requiere_datos
def test_pipeline_sobre_muestra_real_de_test():
    from richter_predictor.data_loader import cargar_test
    test = cargar_test().head(500)
    pred = predecir_dano_estructural(test)
    assert len(pred) == 500 and pred.damage_grade.isin([1, 2, 3]).all()


@pytest.mark.skipif(not RUTA_SUBMISSION.exists(), reason="no se ha generado artifacts/predictions/submission_M5.csv")
def test_archivo_de_envio():
    envio = pd.read_csv(RUTA_SUBMISSION)
    assert list(envio.columns) == ["building_id", "damage_grade"]
    assert len(envio) == 86_868
    assert envio.building_id.is_unique
    assert not envio.isna().any().any()
    assert envio.damage_grade.isin([1, 2, 3]).all()


@requiere_datos
@pytest.mark.skipif(not RUTA_SUBMISSION.exists(), reason="no se ha generado artifacts/predictions/submission_M5.csv")
def test_archivo_de_envio_respeta_submission_format():
    from richter_predictor.data_loader import cargar_submission_format
    formato = cargar_submission_format()
    envio = pd.read_csv(RUTA_SUBMISSION)
    assert (envio.building_id.to_numpy() == formato.building_id.to_numpy()).all()
