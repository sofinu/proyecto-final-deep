# Resultados y verificación

## 1. Procedencia de las cifras

Hay dos fuentes de resultados en este repositorio, que no deben mezclarse:

| Fuente | Qué es | Dónde está |
|---|---|---|
| **Ejecución original del equipo** | Notebook ejecutado en Google Colab (Python 3.13). Son las cifras del informe y de la presentación. | salidas guardadas en `notebooks/proyecto_integrador_richter.ipynb`; tablas extraídas en `reports/tables/` |
| **Reentrenamiento del repositorio** | El mismo M5 reentrenado con `python -m richter_predictor.train` (Python 3.11.9, scikit-learn 1.7.2, numpy 1.26.4) para disponer de un modelo serializado. | `artifacts/models/metricas_m5.json` |

Las cifras oficiales del proyecto son las de la ejecución original. El modelo serializado es una
reproducción con la misma receta, cuyas métricas difieren en el cuarto decimal (sección 3).

## 2. Verificación de las cifras de la presentación

Cada valor se buscó en las salidas del notebook (celda y tabla indicadas).

| Cifra de la presentación | Valor en el notebook | Fuente | Estado |
|---|---|---|---|
| 260,601 edificaciones de train | 260,601 | T01, celda 2.1 | ✅ coincide |
| 86,868 edificaciones de test | 86,868 | T01 | ✅ |
| 38 predictoras originales | 38 | T02 (`Total de predictoras: 38`) | ✅ (el texto de la Sección 2.2 dice 39, ver §4) |
| 99 variables tras el preprocesamiento | 99 | composición del parquet, celda 3 | ✅ |
| TE out-of-fold con 5 folds | `StratifiedKFold(5, shuffle=True, random_state=42)` | `te_oof`, Sección 3.4 | ✅ |
| Fuga en el encoding congelado | hipótesis A reproduce el parquet (dif. 3e-14); correlación 1.0 en zonas de 1 edificio | T08, T09 | ✅ |
| M5: 3 × MLP (64, 32), Adam, early stopping, α = 1e-5, semillas 42/43/44, promedio | código de las celdas de la Parte 2 | celda M5 | ✅ (M5 se construyó sobre M1, sin oversampling ni ordinal) |
| F1 micro 0.7429 | 0.7429 | escalera / T17, val_B | ✅ |
| F1 macro 0.6901 | 0.6901 | T17, val_B | ✅ |
| Recall clase 1 0.5145 | 0.5145 | T17, val_B | ✅ |
| Precisión clase 1 0.6852 | 0.6852 | T17, val_B | ✅ |
| Log-loss 0.5748 | 0.5748 | T17, val_B | ✅ |
| **"Brecha val_A → val_B −0.0002"** | −0.0002 es la **brecha F1 micro train − val_B** | T17, columna `Brecha F1 (train-val_B)` | ⚠️ **etiqueta incorrecta** (ver §4) |
| M2 descartado: recall c1 0.745, F1 micro 0.7256 | 0.7449 y 0.7256 | escalera | ✅ |
| LightGBM F1 micro 0.7474, recall c1 0.5404 | 0.7474 y 0.5404 | celda de referencia LightGBM, val_B | ✅ |
| Pipeline: F1 micro 0.7405 en 2,000 registros | 0.7405 | Sección 6.2 | ✅ (la muestra sale de toda la validación, incluida val_A) |
| 86,868 predicciones de test | 86,868 filas, ids en el orden de `submission_format` | Sección 6.3 | ✅ |

Todas las métricas de M5 pertenecen al modelo correcto (ensamble de la celda M5), al conjunto
indicado (val_B, 26,061 filas) y a la versión de preprocesamiento correcta (TE OOF). No se
mezclan experimentos: la tabla T17 se construye a partir de la misma fila `fila_M5` de la escalera.

## 3. Reproducción local del modelo M5

**Preprocesamiento.** El paquete `src/richter_predictor` produce una matriz de entrenamiento y
validación (float32) **idéntica bit a bit** a la que usó el notebook: 0 celdas distintas de
25,799,499. También reproduce el split original y el parquet histórico (dif. ≤ 4e-15).

**Entrenamiento.** Con esa misma matriz, la misma receta y las mismas semillas, el entrenamiento
local no produce exactamente los mismos pesos: la red con semilla 42 se detiene en la época 13
(en el notebook, la misma configuración en M1 se detuvo en la 15). La causa es el entorno
(versión de Python, numpy y scikit-learn de Colab frente al entorno local; la versión exacta de
Colab no quedó registrada), no los datos.

| Métrica (val_B) | Notebook (Colab) | Reentrenamiento local | Diferencia |
|---|---:|---:|---:|
| F1 micro | 0.7429 | 0.7426 | −0.0003 |
| F1 macro | 0.6901 | 0.6902 | +0.0001 |
| Recall clase 1 | 0.5145 | 0.5201 | +0.0056 |
| Precisión clase 1 | 0.6852 | 0.6783 | −0.0069 |
| Log-loss | 0.5748 | 0.5742 | −0.0006 |
| Brecha F1 train − val_B | −0.0002 | +0.0012 | +0.0014 |
| F1 micro pipeline, 2,000 filas de val | 0.7405 | 0.7435 | +0.0030 |
| Épocas (semillas 42 / 43 / 44) | no registradas | 13 / 12 / 17 | — |
| Distribución predicha en test (1 / 2 / 3, %) | 7.26 / 64.68 / 28.06 | 7.34 / 64.64 / 28.02 | < 0.1 pp |

Las diferencias son del orden del error estándar del F1 micro en val_B (≈ 0.0027) y no cambian
ninguna conclusión. Las del recall y la precisión de la clase 1 son mayores porque esa clase solo
tiene 2,513 casos en val_B. Las métricas del reentrenamiento se guardan en
`artifacts/models/metricas_m5.json`.

## 4. Discrepancias encontradas

No se modificó el texto del notebook, del informe ni de la presentación; se recomienda que el
equipo corrija estos puntos antes de la entrega final.

| # | Dónde | Discrepancia | Corrección sugerida |
|---|---|---|---|
| 1 | Presentación, diapositiva 6 y sus notas | "Brecha val_A → val_B −0.0002". El valor es la brecha F1 micro **train − val_B**; el notebook no calcula el F1 de M5 en val_A. | Cambiar la etiqueta a "Brecha train → val_B". |
| 2 | Presentación, diapositiva 7 | Dice que la distribución predicha en test "se mantiene razonable … sin señales de desajuste grave". El notebook y el informe (Tabla 6) señalan +7.79 pp en la clase 2 y lo tratan como la principal limitación. | Alinear el mensaje con el informe. |
| 3 | Presentación, diapositiva 10 | "Repositorio de código: [enlace de GitHub]" sin completar. | Poner el enlace cuando se publique. |
| 4 | Notebook, Sección 2 (descripción) y lectura de 2.2 | Dicen "39 variables predictoras"; la salida de la celda y el informe dicen 38. | 38 predictoras (39 columnas con `building_id`). |
| 5 | Notebook, lectura de la Tabla T10 (Sección 3.5) | El texto cita brecha 0.0183 → 0.0015, F1 val 0.7367 → 0.7394, F1 macro 0.6814 → 0.6841, log-loss 0.5913 → 0.5839; la tabla guardada muestra 0.0188 → 0.0018, 0.7366 → 0.7395, 0.6808 → 0.6830, 0.5907 → 0.5837. El párrafo termina en una frase incompleta ("la decisión de adoptar OOF"). | Actualizar el texto con la tabla de la ejecución final. |
| 6 | Notebook, lectura de M5 y de LightGBM; informe, Tabla 5 | Tiempos de 136 s (M5) y 77 s (LightGBM) en el texto; las salidas dicen 139.6 s y 75 s. | Usar 139.6 s y 75 s. |
| 7 | Notebook, Sección 6 (introducción) | "escalera de ocho intervenciones"; son seis pasos (M0–M5). | "seis pasos". |
| 8 | Notebook, Sección 7 | Cita una "Sección 4.9" que no existe y afirma que el oversampling "hunde F1 macro": en M2 el F1 macro sube (0.6863 → 0.6887); lo que baja es el F1 micro y la precisión de la clase 1. El informe sí lo dice bien. | Corregir la referencia y la afirmación. |
| 9 | Notebook, portada | El título dice "Parte 1" pero el notebook contiene las tres partes. | Se conservó el título original; el README del repositorio describe las tres partes. |
| 10 | Notebook, contadores de ejecución | Algunas celdas finales tienen contadores no consecutivos (70, 72, 74…): se reejecutaron celdas sueltas en Colab. Las salidas guardadas son consistentes entre sí. | Ninguna; la ejecución local completa confirma que el notebook corre en orden. |
| 11 | Informe | Solo existe en `.docx`; no hay PDF final. | Exportar a PDF para la entrega. |

## 5. Métricas del modelo final: qué significan

- **F1 micro 0.7429.** En multiclase equivale a la exactitud: el modelo acierta el nivel de daño
  en el 74.3 % de los edificios de val_B. Es la métrica oficial de la competencia.
- **F1 macro 0.6901.** Promedia el F1 de las tres clases con el mismo peso; queda por debajo del
  micro porque la clase 1 (9.6 % de los datos) se predice peor.
- **Recall clase 1 0.5145.** De cada 100 edificios con daño bajo, el modelo identifica 51; el
  resto los clasifica casi siempre como daño medio. Es la principal limitación.
- **Precisión clase 1 0.6852.** Cuando predice daño bajo, acierta 69 de cada 100 veces.
- **Log-loss 0.5748.** Mide la calidad de las probabilidades; el ensamble es el paso que más la
  reduce (de 0.5788 en M1 a 0.5748).
- **Brecha F1 train − val_B −0.0002.** No hay sobreajuste: el modelo rinde igual en train que en
  datos no vistos.

Por clase en val_B (M5): recall 0.5145 / 0.8426 / 0.6393 y precisión 0.6852 / 0.7443 / 0.7547
para las clases 1 / 2 / 3. Los errores entre clases no vecinas (1↔3) son el 0.28 % de las filas.
