# Datos

Los datos **no se incluyen en el repositorio**. Pertenecen a la competencia
[Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/)
de DrivenData. Se descargan con una cuenta aceptando las reglas de la competencia, que regulan
su uso; por eso el repositorio no los redistribuye y cada persona debe obtenerlos en la fuente
oficial. Cualquier archivo derivado fila a fila de ellos (por ejemplo, los parquet procesados,
que incluyen `damage_grade`) también queda fuera de Git.

Origen de los datos: encuestas posteriores al sismo de Gorkha (2015) recogidas por Kathmandu
Living Labs y la Central Bureau of Statistics de Nepal.

## 1. Cómo obtenerlos

1. Crea una cuenta en DrivenData e inscríbete en la competencia.
2. En la pestaña **Data** descarga los cuatro archivos:
   `train_values.csv`, `train_labels.csv`, `test_values.csv` y `submission_format.csv`.
3. Cópialos, sin renombrarlos, en `data/raw/`.

## 2. Estructura esperada

```text
data/
├── raw/                                   # NO versionado
│   ├── train_values.csv                   # 260,601 filas × 39 columnas (building_id + 38 predictoras)
│   ├── train_labels.csv                   # 260,601 filas × 2 (building_id, damage_grade)
│   ├── test_values.csv                    #  86,868 filas × 39 columnas, sin etiqueta
│   └── submission_format.csv              #  86,868 filas × 2 (plantilla de envío)
└── processed/
    ├── preprocesamiento_metadata.json     # versionado: orden de las 99 features, categorías, semilla, m
    ├── terremoto_train_procesado.parquet  # NO versionado: preprocesamiento "congelado" histórico
    └── (archivos que genera el notebook)  # NO versionados, ver sección 4
```

Tamaños de referencia de los archivos usados en el proyecto:

| Archivo | Tamaño | Versionado |
|---|---:|:---:|
| `train_values.csv` | 23.4 MB | no |
| `train_labels.csv` | 2.3 MB | no |
| `test_values.csv` | 7.8 MB | no |
| `submission_format.csv` | 0.8 MB | no |
| `terremoto_train_procesado.parquet` | 5.0 MB | no |
| `preprocesamiento_metadata.json` | 7 KB | sí |

Si los CSV están en otra carpeta, se puede indicar con la variable de entorno
`RICHTER_DATA_RAW` (la usan el paquete `richter_predictor` y el notebook).

## 3. Qué necesita cada componente

| Componente | Necesita |
|---|---|
| `python -m richter_predictor.train` | `data/raw/train_values.csv`, `data/raw/train_labels.csv` |
| `python -m richter_predictor.predict` | el CSV a predecir (por defecto `data/raw/test_values.csv`) y `artifacts/models/` |
| `predecir_dano_estructural()` | solo `artifacts/models/` (los artefactos sí están versionados) |
| `notebooks/proyecto_integrador_richter.ipynb` | los 4 CSV de `data/raw/` + `terremoto_train_procesado.parquet` + `preprocesamiento_metadata.json` |
| `pytest` | nada obligatorio; las pruebas que usan datos reales se omiten si faltan |

## 4. El parquet "congelado" y los archivos procesados

`terremoto_train_procesado.parquet` es el dataset que el equipo preprocesó antes del proyecto
integrador (Actividad 5). **El código que lo generó no se conserva**; el notebook lo reconstruye
paso a paso desde los CSV (Sección 3) y comprueba que coincide con diferencias ≤ 4e-15. Esa
auditoría descubrió que su target encoding de `geo_level_2_id`/`geo_level_3_id` incluye la
etiqueta de la propia fila en entrenamiento (fuga), y por eso el modelo final usa la versión
out-of-fold.

El notebook necesita el parquet para la auditoría de las Secciones 2–3. Si no lo tienes, pídelo
al equipo. El pipeline de `src/` **no** depende de él: recalcula todo desde los CSV, y la prueba
`tests/test_preprocessing.py::test_reproduce_split_y_parquet_original` verifica que ambos
coinciden cuando el parquet está disponible.

Al ejecutarse, el notebook escribe en `data/processed/`:

| Archivo | Sección | Contenido |
|---|---|---|
| `terremoto_train_val_te_oof.parquet` | 4 | train + val con el target encoding OOF (entrada de las Partes 2 y 3) |
| `terremoto_test_procesado_te_oof.parquet` | 4 | test transformado con la escala OOF |
| `terremoto_test_procesado_congelado.parquet` | 4 | test con la escala del encoding congelado (solo comparación) |
| `parametros_preprocesamiento.json` | 4 | medias, desviaciones y parámetros del TE |
| `submission_M5.csv` | 6.3 | predicciones del notebook sobre test |

Para regenerarlos basta con ejecutar el notebook completo (ver el README principal).
