"""Entrena el modelo final M5 desde los CSV originales y guarda los artefactos.

Uso:
    python -m richter_predictor.train

Pasos (los mismos del notebook, sin la escalera completa M0-M4):
1. Carga train_values + train_labels y reproduce el split 80/20 (semilla 42).
2. Ajusta el preprocesamiento con train (target encoding OOF en train).
3. Parte validación en val_A / val_B (50/50, semilla 42).
4. Entrena 3 MLP (64, 32), alpha = 1e-5, semillas 42/43/44, early stopping en val_A.
5. Reporta métricas en val_B y guarda preprocesador, ensamble y métricas en artifacts/models/.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import f1_score

from . import config
from .data_loader import cargar_train
from .evaluation import dividir_val_A_B, metricas
from .model import entrenar_m5, guardar_modelo
from .preprocessing import PreprocesadorRichter, mascara_validacion

# Valores de referencia de la ejecución original del notebook (Colab), Tabla T17
REFERENCIA_NOTEBOOK_VAL_B = {"f1_micro": 0.7429, "f1_macro": 0.6901, "recall_clase_1": 0.5145,
                             "precision_clase_1": 0.6852, "log_loss": 0.5748, "brecha_f1_train_menos_val_B": -0.0002}


def preparar_datos(dir_raw: Path | None = None):
    datos = cargar_train(dir_raw)
    es_val = mascara_validacion(datos)
    es_train = ~es_val
    y = datos[config.OBJETIVO].to_numpy()

    pre = PreprocesadorRichter()
    X_train = pre.fit_transform_oof(datos.loc[es_train, config.COLUMNAS_ENTRADA],
                                    datos.loc[es_train, config.OBJETIVO])
    X_val = pre.transform(datos.loc[es_val, config.COLUMNAS_ENTRADA])
    es_A, es_B = dividir_val_A_B(es_val, y)
    return datos, pre, es_train, es_val, es_A, es_B, X_train, X_val


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrena el ensamble M5 y guarda los artefactos.")
    parser.add_argument("--datos", type=Path, default=None, help="carpeta con los CSV (por defecto data/raw)")
    parser.add_argument("--salida", type=Path, default=config.DIR_MODELOS, help="carpeta de artefactos")
    args = parser.parse_args(argv)

    t0 = time.time()
    datos, pre, es_train, es_val, es_A, es_B, X_train_df, X_val_df = preparar_datos(args.datos)
    y = datos[config.OBJETIVO].to_numpy()

    # Matrices float32 en el orden original de filas (como en el notebook)
    X_train = X_train_df.to_numpy(np.float32)
    y_train = y[es_train]
    X_val_all = np.empty((len(datos), X_val_df.shape[1]), dtype=np.float32)
    X_val_all[es_val] = X_val_df.to_numpy(np.float32)
    X_A, y_A = X_val_all[es_A], y[es_A]
    X_B, y_B = X_val_all[es_B], y[es_B]
    print(f"Datos listos en {time.time() - t0:.1f} s · train {len(y_train):,} · val_A {len(y_A):,} · "
          f"val_B {len(y_B):,} · {X_train.shape[1]} features")

    t1 = time.time()
    ens = entrenar_m5(X_train, y_train, X_A, y_A, verbose=True)
    t_entreno = time.time() - t1

    m_B = metricas(y_B, ens.predict_proba(X_B))
    m_A = metricas(y_A, ens.predict_proba(X_A))
    f1_train = float(f1_score(y_train, ens.predict(X_train), average="micro"))
    m_B["brecha_f1_train_menos_val_B"] = f1_train - m_B["f1_micro"]

    # Control del pipeline: la misma muestra de 2,000 filas de validación de la Sección 6.2
    from .pipeline import predecir_dano_estructural
    muestra = datos[es_val].sample(2000, random_state=config.SEMILLA)
    pred = predecir_dano_estructural(muestra[config.COLUMNAS_ENTRADA], modelo=ens, preprocesador=pre)
    f1_muestra = float(f1_score(muestra[config.OBJETIVO].to_numpy(), pred[config.OBJETIVO].to_numpy(),
                                average="micro"))

    args.salida.mkdir(parents=True, exist_ok=True)
    pre.guardar(args.salida / config.ARCHIVO_PREPROCESADOR)
    guardar_modelo(ens, args.salida / config.ARCHIVO_MODELO_M5)
    reporte = {
        "modelo": "M5 · ensamble de 3 MLPClassifier (64, 32), alpha = 1e-5, semillas 42/43/44",
        "val_B": m_B,
        "val_A": m_A,
        "f1_micro_train": f1_train,
        "f1_micro_pipeline_muestra_2000_val": f1_muestra,
        "epocas_por_semilla": {k: v["epoca_mejor"] for k, v in ens.info["por_semilla"].items()},
        "log_loss_val_A_por_semilla": {k: v["val_loss"] for k, v in ens.info["por_semilla"].items()},
        "tiempo_entrenamiento_s": round(t_entreno, 1),
        "referencia_notebook_val_B": REFERENCIA_NOTEBOOK_VAL_B,
        "diferencia_vs_notebook_val_B": {k: round(m_B[k] - v, 4) for k, v in REFERENCIA_NOTEBOOK_VAL_B.items()},
        "n": {"train": int(es_train.sum()), "val_A": int(es_A.sum()), "val_B": int(es_B.sum())},
        "entorno": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__},
    }
    (args.salida / config.ARCHIVO_METRICAS_M5).write_text(json.dumps(reporte, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")

    print("\nMétricas M5 en val_B (esta ejecución vs. notebook original):")
    for k, v in REFERENCIA_NOTEBOOK_VAL_B.items():
        print(f"  {k:<30} {m_B[k]:.4f}   (notebook {v:.4f})")
    print(f"  F1 micro del pipeline en 2,000 filas de val: {f1_muestra:.4f} (notebook 0.7405)")
    print(f"\nArtefactos guardados en {args.salida} · tiempo total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
