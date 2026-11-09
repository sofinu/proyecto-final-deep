# Contribuciones del equipo

Equipo: **Ana Ferrer**, **Sofía Nuñez** y **Valentina Ramírez** — Fundamentos de Redes
Neuronales, Analítica y Ciencia de Datos, Universidad de La Sabana.

## Fuente de esta atribución

El trabajo se hizo en un notebook compartido de Google Colab, sin control de versiones, así que
**no hay historial de commits por persona**. La distribución siguiente se basa en:

1. la división en tres partes acordada por el equipo para el proyecto integrador;
2. la estructura del notebook consolidado (Parte 1: Secciones 0–5; Parte 2: escalera M0–M5 y
   LightGBM; Parte 3: Secciones 6–8), donde cada bloque continúa el trabajo del anterior;
3. la portada del notebook, el informe técnico y la presentación, que firman las tres integrantes.

Si alguna integrante aportó a otra parte, este documento debe ajustarse.

## Responsabilidades

### Sofía Nuñez — Parte 1: problema, datos y preprocesamiento

- Planteamiento del problema y reformulación de la pregunta (asociación, no causalidad).
- Auditoría de los archivos: integridad de identificadores, nulos, duplicados y perfiles idénticos
  con etiquetas distintas, valor especial `age = 995`.
- Reconstrucción del preprocesamiento "congelado" desde los CSV y verificación numérica.
- Diagnóstico de la fuga del target encoding (cada fila de train incluía su propia etiqueta) y
  propuesta del target encoding out-of-fold de 5 pliegues.
- Auditoría del conjunto de prueba (categorías nuevas, deriva) y función `transformar()`.
- Análisis exploratorio (Figuras 1–5, Tablas T01–T16).

### Ana Ferrer — Parte 2: modelado y optimización

- Protocolo de evaluación val_A / val_B para separar decisiones de reporte.
- Entrenamiento con `partial_fit` y early stopping sobre la log-loss de val_A.
- Escalera de optimización M0 → M5: regularización α, oversampling parcial de la clase 1,
  umbrales ordinales, formulación ordinal con dos sigmoides y ensamble de tres semillas.
- Referencia no neuronal con LightGBM.

### Valentina Ramírez — Parte 3: evaluación e integración

- Selección y resumen del modelo final M5 (Tabla T17).
- Análisis de errores con matrices de confusión por paso.
- Pipeline `predecir_dano_estructural()` y su verificación sobre 2,000 filas de validación.
- Predicción sobre el conjunto de prueba, control de la distribución de clases (Tabla T18) y
  archivo de envío.
- Conclusiones técnicas, recomendaciones y figuras de síntesis para el informe (Figuras 6–10).

## Decisiones tomadas en equipo

| Decisión | Motivo |
|---|---|
| Adoptar el target encoding OOF en todo el modelado | La fuga del encoding congelado inflaba el desempeño en train (brecha F1 0.0188 → 0.0018 con OOF en la comparación de la Tabla T10). |
| No tocar el split 80/20 original (semilla 42) | El target encoding depende de qué filas son train; cambiarlo invalidaría la comparación con la Actividad 5. |
| Separar val_A (decidir) y val_B (reportar) | Evitar reportar una métrica optimizada sobre el mismo conjunto. |
| Descartar el oversampling (M2) | Sube el recall de la clase 1 a 0.7449 pero baja el F1 micro de 0.7415 a 0.7256. |
| Entregar M5 y no LightGBM | LightGBM queda 0.0045 de F1 micro por encima, pero el curso pide un modelo neuronal; se usa como referencia del techo alcanzable. |
| F1 micro como métrica principal, con F1 macro y métricas de la clase 1 | F1 micro es la métrica oficial de la competencia; las otras exponen el desbalance. |

## Organización del repositorio

La consolidación del repositorio (estructura, extracción del código a `src/richter_predictor`,
pruebas, artefactos del modelo y documentación) se hizo a partir del notebook final del equipo,
sin cambiar el modelo ni el preprocesamiento.
