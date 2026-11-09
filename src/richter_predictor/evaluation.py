"""Protocolo de evaluación: val_A (desarrollo) y val_B (reporte), y métricas del notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss, precision_recall_fscore_support,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from . import config


def dividir_val_A_B(es_val: np.ndarray, y: np.ndarray, semilla: int = config.SEMILLA):
    """Parte las filas de validación en dos mitades estratificadas (máscaras booleanas).

    Igual que el notebook: val_A se usa para early stopping y para elegir; val_B solo para reportar.
    """
    val_idx = np.where(es_val)[0]
    idx_A, idx_B = train_test_split(val_idx, test_size=config.VAL_B_SIZE, stratify=np.asarray(y)[es_val],
                                    random_state=semilla)
    es_A = np.zeros(len(es_val), dtype=bool)
    es_B = np.zeros(len(es_val), dtype=bool)
    es_A[idx_A] = True
    es_B[idx_B] = True
    return es_A, es_B


def metricas(y_true, proba, clases=config.CLASES) -> dict:
    """Las métricas de la escalera M0-M5 (F1 micro es la métrica oficial de DrivenData)."""
    clases = np.asarray(clases)
    pred = clases[np.asarray(proba).argmax(1)]
    return {
        "f1_micro": float(f1_score(y_true, pred, average="micro")),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "recall_clase_1": float(recall_score(y_true, pred, labels=[1], average="macro")),
        "precision_clase_1": float(precision_score(y_true, pred, labels=[1], average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=clases)),
    }


def reporte_por_clase(y_true, y_pred) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=list(config.CLASES), zero_division=0)
    return pd.DataFrame({"clase": config.CLASES, "precision": p, "recall": r, "f1": f, "soporte": n})


def matriz_confusion(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(config.CLASES))
    return pd.DataFrame(cm, index=[f"real {c}" for c in config.CLASES],
                        columns=[f"pred {c}" for c in config.CLASES])
