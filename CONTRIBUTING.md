# Cómo colaborar

## Ramas

| Rama | Uso |
|---|---|
| `main` | Versión estable y entregable. Solo recibe cambios por pull request revisado. |
| `feature/datos-sofia` | Parte 1: datos, preprocesamiento, auditoría y EDA (Sofía Nuñez). |
| `feature/modelado-ana` | Parte 2: modelado y optimización (Ana Ferrer). |
| `feature/evaluacion-valentina` | Parte 3: evaluación, pipeline y conclusiones (Valentina Ramírez). |

## Flujo de trabajo

```bash
git switch main && git pull                   # partir siempre de main actualizado
git switch feature/datos-sofia                # o la rama propia
git merge main                                # traer los cambios recientes de main
# ... cambios ...
python -m pytest                              # las pruebas deben pasar
git add <archivos> && git commit -m "Describe el cambio"
git push -u origin feature/datos-sofia
# abrir un pull request hacia main y pedir revisión a otra integrante
```

## Reglas

- **No subir datos.** Los CSV de DrivenData y los parquet van en `data/raw/` y `data/processed/`,
  que `.gitignore` excluye. Antes de cada commit, revisar `git status`.
- **No subir credenciales** (`.env`, tokens, `kaggle.json`).
- **Notebooks**: limpiar salidas pesadas que no aporten evidencia y no versionar
  `.ipynb_checkpoints/`. Si se reejecuta el notebook principal, revisar que las cifras del README
  y de `docs/results.md` sigan siendo ciertas.
- **Modelo**: si se cambia el preprocesamiento o la receta de M5, regenerar los artefactos con
  `python -m richter_predictor.train`, actualizar `docs/results.md` y explicar el cambio en el
  pull request. No sobrescribir `artifacts/models/` sin ese registro.
- **Cifras**: toda métrica que se agregue a la documentación debe salir de una salida
  reproducible (notebook, `metricas_m5.json` o una prueba).
- Código y documentación en español.
