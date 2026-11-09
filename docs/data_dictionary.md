# Diccionario de datos

Fuente: `train_values.csv` (260,601 filas). Las categorías son códigos anonimizados por
DrivenData: las letras no tienen un significado público. Los rangos y cardinalidades se
calcularon sobre el train completo (Tabla T02 del notebook).

## Variable objetivo

| Variable | Valores | Distribución en train |
|---|---|---|
| `damage_grade` | 1 = daño bajo · 2 = daño medio · 3 = daño casi total (ordinal) | 25,124 (9.64 %) · 148,259 (56.89 %) · 87,218 (33.47 %) |

## Predictoras originales (38) y su transformación

| Variable | Tipo | Valores distintos | Rango / categorías | Transformación | Features resultantes |
|---|---|---:|---|---|---|
| `geo_level_1_id` | geográfica | 31 | 0–30 | one-hot | `geo_level_1_id_0` … `geo_level_1_id_30` (31) |
| `geo_level_2_id` | geográfica | 1,414 | 0–1427 | target encoding OOF suavizado (m = 15) + estandarización | `geo_level_2_id_target_enc_s` |
| `geo_level_3_id` | geográfica | 11,595 | 0–12567 | target encoding OOF suavizado (m = 15) + estandarización | `geo_level_3_id_target_enc_s` |
| `count_floors_pre_eq` | numérica | 9 | 1–9 | StandardScaler (train) | `count_floors_pre_eq_s` |
| `age` | numérica | 42 | 0–995 | 995 → mediana de train (15) + bandera; StandardScaler | `age_s`, `age_valor_especial_995` |
| `area_percentage` | numérica | 84 | 1–100 | StandardScaler (train) | `area_percentage_s` |
| `height_percentage` | numérica | 27 | 2–32 | StandardScaler (train) | `height_percentage_s` |
| `count_families` | numérica | 10 | 0–9 | StandardScaler (train) | `count_families_s` |
| `land_surface_condition` | categórica | 3 | n, o, t | one-hot | 3 |
| `foundation_type` | categórica | 5 | h, i, r, u, w | one-hot | 5 |
| `roof_type` | categórica | 3 | n, q, x | one-hot | 3 |
| `ground_floor_type` | categórica | 5 | f, m, v, x, z | one-hot | 5 |
| `other_floor_type` | categórica | 4 | j, q, s, x | one-hot | 4 |
| `position` | categórica | 4 | j, o, s, t | one-hot | 4 |
| `plan_configuration` | categórica | 10 | a, c, d, f, m, n, o, q, s, u | one-hot | 10 |
| `legal_ownership_status` | categórica | 4 | a, r, v, w | one-hot | 4 |
| `has_superstructure_*` (11) | binaria | 2 | 0–1 | sin cambios | 11 |
| `has_secondary_use*` (11) | binaria | 2 | 0–1 | sin cambios | 11 |

Binarias de superestructura: `adobe_mud`, `mud_mortar_stone`, `stone_flag`,
`cement_mortar_stone`, `mud_mortar_brick`, `cement_mortar_brick`, `timber`, `bamboo`,
`rc_non_engineered`, `rc_engineered`, `other`.

Binarias de uso secundario: `has_secondary_use` y `_agriculture`, `_hotel`, `_rental`,
`_institution`, `_school`, `_industry`, `_health_post`, `_gov_office`, `_use_police`, `_other`.

## Las 99 features del modelo

| Bloque | Columnas |
|---|---:|
| One-hot (8 categóricas + `geo_level_1_id`) | 69 |
| Numéricas estandarizadas | 5 |
| Target encoding estandarizado (geo 2 y 3) | 2 |
| Binarias originales | 22 |
| Bandera `age_valor_especial_995` | 1 |
| **Total** | **99** |

El orden exacto está en `data/processed/preprocesamiento_metadata.json` (`feature_cols`) y en
`artifacts/models/preprocesador_m5.json`. El pipeline siempre devuelve las columnas en ese orden.

## Valores especiales y calidad

- **`age = 995`**: 1,390 filas en train (0.53 %) y 461 en test. Las edades reales van de 0 a 200
  en múltiplos de 5; 995 es un código, no una edad. Su distribución de daño también es distinta
  (12.9 % de daño bajo frente a 9.6 % global).
- **Nulos**: ninguno en los cuatro archivos.
- **Duplicados**: no hay `building_id` repetidos, pero 16,191 filas repiten el perfil de otra en
  las 38 predictoras; en 3,825 de esos perfiles (9,624 filas) conviven etiquetas distintas. Se
  conservan: reflejan información que el cuestionario no recoge y fijan un piso de error irreducible.
- **Categorías nuevas en test**: ninguna en las variables de baja cardinalidad; 4 de
  `geo_level_2_id` (5 filas) y 266 de `geo_level_3_id` (327 filas, 0.38 % de test), que reciben la
  media global de train en el target encoding.
- **Deriva train → test**: diferencia máxima de proporción de 0.28 puntos porcentuales y las
  mismas medianas en las cinco numéricas.
