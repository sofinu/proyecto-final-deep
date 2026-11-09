"""Constantes, rutas e hiperparámetros del proyecto.

Todos los valores salen del notebook `notebooks/proyecto_integrador_richter.ipynb`
y de `data/processed/preprocesamiento_metadata.json`. Si se cambia alguno, el
modelo deja de ser el M5 reportado.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- rutas
# El paquete se instala en modo editable (pip install -e .), así que la raíz
# del repositorio está dos niveles por encima de src/richter_predictor/.
RAIZ_REPO = Path(__file__).resolve().parents[2]

DIR_DATOS_RAW = Path(os.environ.get("RICHTER_DATA_RAW", RAIZ_REPO / "data" / "raw"))
DIR_DATOS_PROCESADOS = Path(os.environ.get("RICHTER_DATA_PROCESSED", RAIZ_REPO / "data" / "processed"))
DIR_MODELOS = Path(os.environ.get("RICHTER_MODELS_DIR", RAIZ_REPO / "artifacts" / "models"))
DIR_PREDICCIONES = RAIZ_REPO / "artifacts" / "predictions"

ARCHIVO_TRAIN_VALUES = "train_values.csv"
ARCHIVO_TRAIN_LABELS = "train_labels.csv"
ARCHIVO_TEST_VALUES = "test_values.csv"
ARCHIVO_SUBMISSION_FORMAT = "submission_format.csv"
ARCHIVO_METADATA = "preprocesamiento_metadata.json"

ARCHIVO_PREPROCESADOR = "preprocesador_m5.json"
ARCHIVO_MODELO_M5 = "ensamble_m5.joblib"
ARCHIVO_METRICAS_M5 = "metricas_m5.json"

# ------------------------------------------------------- semilla y split
SEMILLA = 42            # misma semilla del preprocesamiento original (metadata["random_state"])
VAL_SIZE = 0.2          # 80 % train / 20 % val, estratificado por damage_grade
VAL_B_SIZE = 0.5        # val se parte en val_A (desarrollo) y val_B (reporte)

# ------------------------------------------------------------- variables
ID = "building_id"
OBJETIVO = "damage_grade"
CLASES = (1, 2, 3)
NOMBRES_CLASE = {1: "daño bajo", 2: "daño medio", 3: "daño casi total"}

# Orden de columnas de train_values.csv / test_values.csv (38 predictoras + id)
COLUMNAS_ENTRADA = [
    "building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id", "count_floors_pre_eq", "age",
    "area_percentage", "height_percentage", "land_surface_condition", "foundation_type", "roof_type",
    "ground_floor_type", "other_floor_type", "position", "plan_configuration",
    "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone", "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone", "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick", "has_superstructure_timber", "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered", "has_superstructure_other",
    "legal_ownership_status", "count_families", "has_secondary_use", "has_secondary_use_agriculture",
    "has_secondary_use_hotel", "has_secondary_use_rental", "has_secondary_use_institution",
    "has_secondary_use_school", "has_secondary_use_industry", "has_secondary_use_health_post",
    "has_secondary_use_gov_office", "has_secondary_use_use_police", "has_secondary_use_other",
]
PREDICTORAS = [c for c in COLUMNAS_ENTRADA if c != ID]

NUMERICAS = ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"]
CAT_BAJA_CARDINALIDAD = ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
                         "other_floor_type", "position", "plan_configuration", "legal_ownership_status",
                         "geo_level_1_id"]
CAT_ALTA_CARDINALIDAD = ["geo_level_2_id", "geo_level_3_id"]
BINARIAS = [c for c in PREDICTORAS if c.startswith("has_")]
COLUMNAS_ENTERAS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"] + NUMERICAS + BINARIAS

# ---------------------------------------------------------- preprocesamiento
VALOR_ESPECIAL_AGE = 995        # código no documentado: bandera + imputación con la mediana de train
SMOOTHING_M = 15                # suavizado del target encoding
N_FOLDS_OOF = 5                 # pliegues del target encoding out-of-fold

# ------------------------------------------------------- modelo final (M5)
HIDDEN_LAYERS = (64, 32)
ALPHA_M5 = 1e-5                 # elegido en M1 por F1 micro en val_A
LEARNING_RATE = 1e-3
MAX_EPOCHS = 80
PATIENCE = 8                    # épocas sin mejora de log-loss en val_A antes de parar
TOL_MEJORA = 1e-4               # mejora mínima de log-loss que cuenta como mejora
SEMILLAS_M5 = (42, 43, 44)
