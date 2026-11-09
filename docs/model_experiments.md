# Experimentos: escalera de optimización M0 → M5

Todas las cifras provienen de las salidas guardadas del notebook (ejecución original en Colab,
celdas de la Parte 2). Se evalúan sobre **val_B** (26,061 edificios), que no se usó para tomar
ninguna decisión. Cada paso parte del mejor modelo aceptado hasta ese momento.

## Configuración común

| Elemento | Valor |
|---|---|
| Red | `MLPClassifier`, capas ocultas (64, 32), ReLU, salida softmax de 3 clases |
| Optimizador | Adam, `learning_rate_init = 1e-3`, lotes de 200 (valor por defecto) |
| Entrenamiento | `partial_fit` época a época, copia con menor log-loss en val_A |
| Early stopping | mejora mínima 1e-4, paciencia 8, máximo 80 épocas |
| Datos | 208,480 filas de train con TE OOF, 99 features (float32) |
| Semilla | 42 (M5 usa 42, 43 y 44) |

## Resultados (val_B)

| Paso | Intervención | F1 micro | F1 macro | Recall c1 | Precisión c1 | Log-loss | Brecha F1 train−val_B | Época | Decisión |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0 | Base OOF + early stopping, α = 1e-4 | 0.7403 | 0.6824 | 0.4934 | 0.6681 | 0.5801 | −0.0012 | 10 | base |
| M1 | Grid α ∈ {1e-5, 1e-4, 1e-3, 1e-2} → α = 1e-5 | 0.7415 | 0.6863 | 0.5066 | 0.6718 | 0.5788 | −0.0004 | 15 | aceptado |
| M2 | Oversampling ×2 de la clase 1 | 0.7256 | 0.6887 | 0.7449 | 0.5263 | 0.6065 | 0.0017 | 9 | rechazado |
| M3 | Umbrales ordinales (t₂, t₃) sobre M1 | 0.7413 | 0.6830 | 0.4851 | 0.6856 | 0.5788 | −0.0009 | 15 | no mejora en val_A |
| M4 | Ordinal: dos sigmoides P(y≥2), P(y≥3) | 0.7411 | 0.6846 | 0.4950 | 0.6757 | 0.5825 | 0.0002 | 7 | no mejora en val_A |
| **M5** | **Ensamble de 3 semillas (42, 43, 44) sobre M1** | **0.7429** | **0.6901** | **0.5145** | **0.6852** | **0.5748** | **−0.0002** | — | **modelo final** |
| Ref. | LightGBM (802 árboles) sobre las mismas features | 0.7474 | 0.6983 | 0.5404 | 0.6848 | — | — | — | solo referencia |

La selección se hace con F1 micro en **val_A**, no en val_B. Grid de α en val_A (M1):

| α | F1 micro val_A | F1 macro val_A | Época |
|---|---:|---:|---:|
| 1e-5 | 0.7380 | 0.6833 | 15 |
| 1e-4 | 0.7371 | 0.6811 | 10 |
| 1e-3 | 0.7367 | 0.6810 | 14 |
| 1e-2 | 0.7373 | 0.6831 | 12 |

## Lectura de cada paso

- **M0.** Sin brecha train−validación. El recall de la clase 1 (0.49) indica que el modelo es
  conservador al predecir daño bajo.
- **M1.** Los cuatro α quedan en un rango de 0.0013 de F1 micro: con early stopping y sin
  sobreajuste, la regularización L2 no es una palanca relevante. Se adopta 1e-5 por ser el mejor
  en val_A.
- **M2.** Duplicar la clase 1 sube su recall de 0.507 a 0.745 pero baja la precisión a 0.526 y el
  F1 micro en 0.0159: se ganan unos 600 aciertos de daño bajo y se pierden unos 1,000 en las otras
  clases. El early stopping de este paso pondera ×2 la clase 1 de val_A para que tenga la misma
  mezcla que el train sobremuestreado. Se descarta porque perjudica la métrica principal.
- **M3.** El grid de umbrales (paso 0.05 en [0.2, 0.8]) con criterio F1 micro termina en
  (0.50, 0.50) y no supera el argmax en val_A. Optimizar F1 micro no recupera la clase 1.
- **M4.** La formulación ordinal empata con M1 en F1 micro y empeora la log-loss. No se adopta.
- **M5.** Promediar las probabilidades de tres redes es el único paso posterior a M1 que mejora
  todas las métricas de val_B. La mejora más clara es la log-loss (−0.0040), lo esperable de un
  ensamble. Costo: unas 2.7 veces el tiempo de entrenamiento.

## Alcance de las mejoras

De M0 a M5 el F1 micro sube 0.0026, del mismo orden que su error estándar en val_B (≈ 0.0027).
Con una sola corrida por paso, ninguna diferencia individual es distinguible del ruido de
muestreo. Lo que sí sostienen los datos: la regularización y los umbrales no son palancas
relevantes, el oversampling es la única intervención que cambia el balance entre clases (a costa
del F1 micro) y el ensamble mejora de forma consistente la log-loss.

En todos los pasos casi todos los errores son entre clases vecinas: los errores 1↔3 son alrededor
del 1 % de los errores (0.28 % de las filas de val_B en M5).

## Referencia LightGBM

`LGBMClassifier` (multiclase, `learning_rate = 0.05`, `num_leaves = 63`, early stopping 50 sobre
val_A, 802 iteraciones) entrenado con las mismas 99 features. Supera a M5 por 0.0045 de F1 micro
(≈ 1.7 errores estándar), con la diferencia concentrada en el recall de la clase 1 (+0.026) y la
misma precisión. Se interpreta como evidencia de que M5 está cerca del techo alcanzable con estas
variables; no reemplaza al modelo entregable, que debía ser una red neuronal.
