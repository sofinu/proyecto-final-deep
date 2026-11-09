# Presentación ejecutiva

| Archivo | Descripción |
|---|---|
| `presentacion_ejecutiva.pptx` | 10 diapositivas con notas del orador, pensadas para un video de 8–10 minutos. |
| Video | **Pendiente.** El MP4 no está en el repositorio; cuando se publique (por ejemplo, en YouTube o Drive), agregar el enlace aquí. |

## Contenido

1. Portada · 2. Problema · 3. Dataset y preprocesamiento · 4. Decisiones técnicas ·
5. Escalera M0 → M5 · 6. Resultado final M5 · 7. Comparación con LightGBM y control en test ·
8. Pipeline integrado · 9. Conclusiones · 10. Cierre.

## Correspondencia con el repositorio

Las cifras de la presentación coinciden con las salidas del notebook
(`notebooks/proyecto_integrador_richter.ipynb`) y con `reports/tables/T17_modelo_final_seleccionado.csv`.
Antes de grabar o entregar conviene corregir tres puntos (detalle en `docs/results.md`, §4):

- Diapositiva 6: la métrica "Brecha val_A → val_B −0.0002" es en realidad la brecha
  **train − val_B**.
- Diapositiva 7: el texto dice que la distribución predicha en test "se mantiene razonable",
  mientras que el informe reporta +7.79 pp en la clase 2 como limitación principal.
- Diapositiva 10: falta el enlace del repositorio: https://github.com/sofinu/proyecto-final-deep
