"""Genera predicciones con el ensamble M5 guardado.

Uso:
    python -m richter_predictor.predict --entrada data/raw/test_values.csv \
        --salida artifacts/predictions/submission_M5.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .pipeline import predecir_dano_estructural


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predice damage_grade con el ensamble M5.")
    parser.add_argument("--entrada", type=Path, default=config.DIR_DATOS_RAW / config.ARCHIVO_TEST_VALUES)
    parser.add_argument("--salida", type=Path, default=config.DIR_PREDICCIONES / "submission_M5.csv")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.entrada)
    pred = predecir_dano_estructural(df)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.salida, index=False)

    dist = (pred[config.OBJETIVO].value_counts(normalize=True).sort_index() * 100).round(2)
    print(f"{len(pred):,} predicciones guardadas en {args.salida}")
    print("Distribución predicha (%):", dist.to_dict())


if __name__ == "__main__":
    main()
