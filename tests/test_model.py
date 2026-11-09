import numpy as np
import pytest

from richter_predictor import config
from richter_predictor.model import EnsambleM5, cargar_modelo, entrenar_mlp_early_stopping, guardar_modelo

from conftest import requiere_artefactos


def _datos_juguete(n=600, d=99, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.7, 0.7]) + 1   # clases 1, 2, 3
    return X, y


def test_entrenamiento_corto_y_ensamble(tmp_path):
    X, y = _datos_juguete()
    modelos = [entrenar_mlp_early_stopping(X[:400], y[:400], X[400:], y[400:], max_epochs=3, semilla=s)[0]
               for s in (42, 43)]
    ens = EnsambleM5(modelos, [42, 43])
    proba = ens.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1)
    assert set(np.unique(ens.predict(X))) <= {1, 2, 3}

    ruta = guardar_modelo(ens, tmp_path / "m.joblib")
    ens2 = cargar_modelo(ruta)
    assert np.allclose(ens2.predict_proba(X), proba)


def test_cargar_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_modelo(tmp_path / "no_existe.joblib")


@requiere_artefactos
def test_modelo_final_m5_se_carga_con_su_configuracion():
    ens = cargar_modelo(config.DIR_MODELOS / config.ARCHIVO_MODELO_M5)
    assert len(ens.modelos) == 3
    assert ens.semillas == [42, 43, 44]
    assert ens.n_features_in_ == 99
    for m, s in zip(ens.modelos, ens.semillas):
        assert m.hidden_layer_sizes == (64, 32)
        assert m.alpha == 1e-5
        assert m.solver == "adam" and m.activation == "relu"
        assert m.random_state == s
        assert list(m.classes_) == [1, 2, 3]


@requiere_artefactos
def test_modelo_final_predice_clases_validas():
    ens = cargar_modelo(config.DIR_MODELOS / config.ARCHIVO_MODELO_M5)
    X = np.random.default_rng(3).normal(size=(200, 99)).astype(np.float32)
    proba = ens.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)
    pred = ens.predict(X)
    assert pred.shape == (200,)
    assert set(np.unique(pred)) <= {1, 2, 3}
