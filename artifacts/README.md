# Artefactos

## `models/`

| Archivo | Tamaño | Contenido |
|---|---:|---|
| `ensamble_m5.joblib` | 0.4 MB | Las 3 redes `MLPClassifier` de M5 (semillas 42, 43, 44), sus hiperparámetros y las versiones del entorno con que se guardaron. |
| `preprocesador_m5.json` | 0.2 MB | Parámetros del preprocesamiento ajustados con train: mediana de `age`, medias y desviaciones del escalado, categorías del one-hot, suma y conteo de `damage_grade` por zona `geo_level_2_id`/`geo_level_3_id`, escalas del TE y orden de las 99 features. |
| `metricas_m5.json` | 2 KB | Métricas del ensamble en val_A, val_B y train, épocas por semilla y comparación con el notebook. |

### Procedencia

Generados el 2026-11-08 con `python -m richter_predictor.train` a partir de `train_values.csv` y
`train_labels.csv`, en Python 3.11.9 con scikit-learn 1.7.2 y numpy 1.26.4 (Windows 11).

**No son los pesos de la ejecución original en Colab.** Esa ejecución no guardó el modelo.
Estos artefactos son un reentrenamiento con la misma receta (misma matriz de entrada, bit a bit,
mismas semillas e hiperparámetros). Sus métricas en val_B difieren de las reportadas en el
cuarto decimal: F1 micro 0.7426 frente a 0.7429 (ver `docs/results.md`, sección 3).

| Semilla | Época elegida | Log-loss val_A |
|---:|---:|---:|
| 42 | 13 | 0.5901 |
| 43 | 12 | 0.5927 |
| 44 | 17 | 0.5911 |

### Uso

```python
from richter_predictor.pipeline import cargar_artefactos, predecir_dano_estructural

preprocesador, modelo = cargar_artefactos()          # lee artifacts/models/
pred = predecir_dano_estructural(df_crudo)           # usa los mismos artefactos por defecto
```

`ensamble_m5.joblib` es un pickle de scikit-learn: se debe cargar con **scikit-learn 1.7.2**
(la versión fijada en `requirements.txt`). Con otra versión, `cargar_modelo` emite una
advertencia. Como todo pickle, solo debe cargarse si proviene de una fuente de confianza.

### Nota sobre los datos

El preprocesador guarda estadísticos agregados de `damage_grade` por zona geográfica (son los
parámetros del target encoding, necesarios para predecir). No contiene filas individuales del
dataset ni `building_id`.

## `predictions/`

| Archivo | Contenido |
|---|---|
| `submission_M5.csv` | 86,868 predicciones sobre `test_values.csv` con el formato de `submission_format.csv` (`building_id`, `damage_grade`), generadas con `python -m richter_predictor.predict`. Distribución: 7.34 % / 64.64 % / 28.02 % para las clases 1 / 2 / 3. |

El `submission_M5.csv` original del notebook se guardó en Colab y no está disponible; este se
regeneró con los artefactos de `models/`. Su desempeño en test solo puede conocerse enviándolo a
DrivenData.
