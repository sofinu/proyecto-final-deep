# Reportes

## Informe técnico

| Archivo | Estado |
|---|---|
| `informe_tecnico_final.docx` | Versión final del informe del proyecto integrador (Unidad 3). Sus cifras coinciden con las salidas del notebook, salvo los tiempos de la Tabla 5 (ver `docs/results.md`, §4). |
| PDF del informe | **Pendiente.** No existe todavía una exportación a PDF. |
| `antecedentes/actividad5_informe_modelo_base.pdf` | Informe de la Actividad 5 (modelo base, previo al proyecto integrador). Se conserva como antecedente; sus cifras corresponden al encoding congelado y no son comparables con M0–M5. |

## `figures/`

Imágenes extraídas de las salidas guardadas del notebook (ejecución original en Colab), con los
nombres que el notebook usa en `savefig`.

| Archivo | Figura |
|---|---|
| `fig1_distribucion_damage_grade.png` | Figura 1 — distribución de la variable objetivo |
| `fig2_antiguedad_por_damage_grade.png` | Figura 2 — antigüedad por nivel de daño |
| `fig3_cimentacion_por_damage_grade.png` | Figura 3 — daño según tipo de cimentación |
| `fig4_geo_level_1_por_damage_grade.png` | Figura 4 — daño por región de primer nivel |
| `fig5_fuga_target_encoding_geo3.png` | Figura 5 — fuga del encoding congelado frente a OOF |
| `fig_matrices_confusion_escalera.png` | Matrices de confusión de M0–M5 en val_B |
| `fig_referencia_arboles.png` | M5 frente a LightGBM en val_B |
| `fig_pipeline_integrado.png` | Figura 6 — pipeline integrado |
| `fig7_clases_informe.png` | Figura 7 (informe) — distribución de clases |
| `fig8_escalera_informe.png` | Figura 8 (informe) — escalera de optimización |
| `fig9_lgbm_test_informe.png` | Figura 9 (informe) — techo de desempeño y salud en test |
| `fig10_pipeline_informe.png` | Figura 10 (informe) — pipeline integrado |

## `tables/`

Tablas recuperadas de las salidas HTML del notebook con `python scripts/extraer_tablas_notebook.py`
(no se reejecutó nada). Corresponden a las cifras reportadas.

| Archivos | Contenido |
|---|---|
| `T01`–`T09` | Auditoría de archivos, diccionario, clases, duplicados, numéricas, split, escalado, reconstrucción y fuga del target encoding |
| `T10` | Impacto de la fuga: encoding congelado frente a OOF |
| `T11` | Resumen de transformaciones |
| `T12`, `T13` | Compatibilidad y deriva del conjunto de prueba |
| `T14`–`T16` | Tablas del análisis exploratorio |
| `M1_grid_alpha_val_A` | Barrido de α en val_A |
| `tabla2_escalera_optimizacion` | Escalera M0–M5 en val_B |
| `errores_por_clase_escalera_val_B` | Recall y precisión por clase de cada paso |
| `M5_vs_LightGBM_val_B` | Comparación con la referencia de árboles |
| `T17_modelo_final_seleccionado` | Métricas del modelo final |
| `T18_distribucion_clases_test_vs_train` | Distribución predicha en test frente a la real de train |

Si se ejecuta el notebook completo, las figuras y tablas de esta carpeta se sobrescriben con los
resultados de esa ejecución (que pueden variar en el cuarto decimal según el entorno).
