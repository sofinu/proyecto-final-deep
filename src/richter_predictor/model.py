"""Modelo final M5: ensamble de 3 MLPClassifier (64, 32) con semillas 42, 43 y 44.

`entrenar_mlp_early_stopping` es la función `entrenar_early_stopping` del notebook
(Sección "Parte 2"): entrena época a época con `partial_fit` y conserva la copia
con menor log-loss en val_A (paciencia 8, máximo 80 épocas).
"""
from __future__ import annotations

import copy
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from . import config


def entrenar_mlp_early_stopping(X_tr, y_tr, X_va, y_va, hidden=config.HIDDEN_LAYERS, alpha=config.ALPHA_M5,
                                lr=config.LEARNING_RATE, max_epochs=config.MAX_EPOCHS,
                                patience=config.PATIENCE, semilla=config.SEMILLA, verbose=False):
    """Devuelve (mejor_modelo, {"epoca_mejor", "val_loss"})."""
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        alpha=alpha, learning_rate_init=lr, max_iter=1, warm_start=True,
                        random_state=semilla)
    clases = np.unique(y_tr)
    clf.partial_fit(X_tr, y_tr, classes=clases)
    mejor, mejor_loss = copy.deepcopy(clf), log_loss(y_va, clf.predict_proba(X_va), labels=clases)
    sin_mejora, epoca_mejor = 0, 1
    for ep in range(2, max_epochs + 1):
        clf.partial_fit(X_tr, y_tr, classes=clases)
        vl = log_loss(y_va, clf.predict_proba(X_va), labels=clases)
        if vl < mejor_loss - config.TOL_MEJORA:
            mejor_loss, mejor, sin_mejora, epoca_mejor = vl, copy.deepcopy(clf), 0, ep
        else:
            sin_mejora += 1
            if sin_mejora >= patience:
                break
    if verbose:
        print(f"  semilla {semilla}: early stop en época {epoca_mejor} (log-loss val_A = {mejor_loss:.4f})")
    return mejor, {"epoca_mejor": epoca_mejor, "val_loss": float(mejor_loss)}


class EnsambleM5:
    """Promedia `predict_proba` de varias redes y predice con argmax (clases 1, 2, 3)."""

    def __init__(self, modelos, semillas=None, info=None):
        if not modelos:
            raise ValueError("El ensamble necesita al menos un modelo.")
        self.modelos = list(modelos)
        self.semillas = list(semillas) if semillas is not None else None
        self.info = info or {}
        self.classes_ = np.array(config.CLASES)
        for m in self.modelos:
            if not np.array_equal(m.classes_, self.classes_):
                raise ValueError(f"Un modelo del ensamble tiene clases {m.classes_}, se esperaban {self.classes_}.")

    @property
    def n_features_in_(self) -> int:
        return int(self.modelos[0].n_features_in_)

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.modelos], axis=0)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]


def entrenar_m5(X_tr, y_tr, X_va, y_va, semillas=config.SEMILLAS_M5, alpha=config.ALPHA_M5, verbose=True):
    modelos, info = [], {}
    for semilla in semillas:
        m, h = entrenar_mlp_early_stopping(X_tr, y_tr, X_va, y_va, alpha=alpha, semilla=semilla, verbose=verbose)
        modelos.append(m)
        info[int(semilla)] = h
    return EnsambleM5(modelos, semillas, {"por_semilla": info})


def guardar_modelo(ensamble: EnsambleM5, ruta: Path) -> Path:
    """Serializa las redes (no la clase del ensamble) junto con su procedencia."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "modelos": ensamble.modelos,
        "semillas": ensamble.semillas,
        "info": ensamble.info,
        "hiperparametros": {"hidden_layer_sizes": config.HIDDEN_LAYERS, "activation": "relu", "solver": "adam",
                            "alpha": config.ALPHA_M5, "learning_rate_init": config.LEARNING_RATE,
                            "max_epochs": config.MAX_EPOCHS, "patience": config.PATIENCE,
                            "criterio_parada": "log-loss en val_A"},
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__},
    }, ruta, compress=3)
    return ruta


def cargar_modelo(ruta: Path) -> EnsambleM5:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el modelo {ruta}. Entrénalo con `python -m richter_predictor.train` (ver README)."
        )
    datos = joblib.load(ruta)
    if datos["versiones"]["scikit-learn"] != sklearn.__version__:
        import warnings
        warnings.warn(f"El modelo se guardó con scikit-learn {datos['versiones']['scikit-learn']} y se está "
                      f"cargando con {sklearn.__version__}; las predicciones podrían variar.")
    ens = EnsambleM5(datos["modelos"], datos["semillas"], datos["info"])
    ens.hiperparametros = datos["hiperparametros"]
    ens.versiones = datos["versiones"]
    return ens
