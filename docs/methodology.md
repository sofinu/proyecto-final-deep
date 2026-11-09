# Metodología

Este documento resume el proceso completo. El detalle, con todas las salidas, está en
`notebooks/proyecto_integrador_richter.ipynb`; el número de sección entre paréntesis remite a él.

## 1. Pregunta y alcance (Sección 1)

¿Se puede predecir el nivel de daño estructural (1 bajo, 2 medio, 3 casi total) que sufrió un
edificio en el sismo de Gorkha de 2015 a partir de sus características constructivas, materiales
y ubicación? Es una clasificación ordinal de tres clases. Los resultados describen asociaciones
en los datos de 2015, no relaciones causales, y no se validaron para otros sismos.

Métrica principal: **F1 micro**, la métrica oficial de la competencia (en multiclase equivale a la
exactitud). Como la clase 1 es solo el 9.6 %, se reportan también F1 macro, recall y precisión de
la clase 1 y log-loss.

## 2. Auditoría de los datos (Sección 2)

- Integridad: 260,601 filas de train y 86,868 de test; `building_id` único, sin intersección
  entre train y test; train_values y train_labels en el mismo orden; sin nulos.
- 38 predictoras (no 39: `building_id` es un identificador).
- Perfiles repetidos: 16,191 filas duplican otra en las 38 predictoras; en 3,825 perfiles
  conviven etiquetas distintas. Se conservan y se interpretan como error irreducible.
- `age = 995` es un código especial (1,390 filas en train).

## 3. Preprocesamiento y control de fuga (Sección 3)

El equipo partía de un parquet ya procesado (Actividad 5) cuyo código original no se conservó.
Se reconstruyó cada transformación desde los CSV y se comparó numéricamente con el parquet:

| Transformación | Verificación |
|---|---|
| Split 80/20 estratificado, `random_state = 42` | `train_test_split` reproduce exactamente la columna `split` |
| `age = 995` → bandera + mediana de train (15) | bandera idéntica a `age == 995` |
| StandardScaler en 5 numéricas | coincide solo si se ajusta con train (dif. ≤ 2e-11) |
| One-hot de 8 categóricas + `geo_level_1_id` | 0 celdas discrepantes |
| Target encoding suavizado (m = 15) de geo 2 y 3 | coincide con la hipótesis "todas las filas de train, incluida la propia" (dif. ≤ 1e-13) |

**Fuga detectada.** En el encoding original cada fila de train se codifica con una media que
incluye su propia etiqueta. En las 1,078 filas cuya zona `geo_level_3_id` tiene un solo edificio,
el encoding es una función lineal de la etiqueta (correlación 1.0). Validación estaba limpia.

**Corrección.** Target encoding **out-of-fold**: train se divide en 5 pliegues estratificados
(semilla 42) y cada fila se codifica con los estadísticos de los otros 4. Validación y test usan
los estadísticos de todo train. Las dos columnas se estandarizan con la media y desviación del
TE OOF de train. Con la misma red y semilla, la brecha F1 train−val pasó de 0.0188 a 0.0018
(Tabla T10).

## 4. Conjunto de prueba (Sección 4)

Sin categorías nuevas en variables de baja cardinalidad, deriva máxima de 0.28 pp. La función
`transformar()` aplica los parámetros de train a filas crudas y se verificó contra el parquet de
validación (dif. 2.2e-11).

## 5. Análisis exploratorio (Sección 5)

- Desbalance 9.6 / 56.9 / 33.5 %; predecir siempre "daño medio" da F1 micro 0.569 y F1 macro 0.242.
- La mediana de antigüedad crece con el daño (5, 15 y 20 años).
- `foundation_type = r` (84 % de los edificios) concentra el daño casi total (37.8 %).
- La región (`geo_level_1_id`) es la variable de mayor contraste: el daño casi total va de 8.6 %
  a 80.8 % según la región.

## 6. Protocolo de modelado (Parte 2)

- Entrada: 99 features con TE OOF, en float32.
- Validación (52,121 filas) se parte en **val_A** (26,060) para early stopping, selección de α y
  umbrales, y **val_B** (26,061) solo para reportar. Ambas estratificadas, semilla 42.
- Entrenamiento época a época con `partial_fit`; se guarda la copia con menor log-loss en val_A
  (mejora mínima 1e-4, paciencia 8, máximo 80 épocas).
- Una intervención a la vez sobre el mejor modelo aceptado (escalera M0 → M5, ver
  [model_experiments.md](model_experiments.md)).

## 7. Evaluación e integración (Parte 3)

- Modelo final: M5, ensamble de 3 MLP. Métricas en val_B, matrices de confusión y comparación con
  LightGBM entrenado sobre las mismas features.
- Pipeline de extremo a extremo: `transformar()` + ensamble → `predecir_dano_estructural()`.
  Verificado sobre 2,000 filas de validación y aplicado a las 86,868 filas de test.
- Control de salud: distribución de clases predicha en test frente a la real de train (Tabla T18).

## 8. Del notebook al paquete

El paquete `src/richter_predictor` extrae ese mismo flujo para que no dependa del estado de un
notebook: `preprocessing.py` (Secciones 3–4), `model.py` (función de early stopping y ensamble),
`evaluation.py` (val_A/val_B y métricas) y `pipeline.py` (Sección 6.2). La matriz de entrenamiento
que produce es idéntica bit a bit a la del notebook (0 celdas distintas de 25.8 millones).
