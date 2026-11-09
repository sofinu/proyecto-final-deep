"""Preprocesamiento de las 38 variables originales a las 99 features del modelo.

Reproduce las Secciones 3 y 4 del notebook (funciones `num_base`, `te_estadisticos`,
`te_aplicar`, `te_oof` y `transformar`). Todos los parámetros se ajustan solo con
el split de entrenamiento:

- age = 995 -> bandera `age_valor_especial_995` + imputación con la mediana de train.
- 5 numéricas -> StandardScaler (media y desviación poblacional de train).
- 8 categóricas + geo_level_1_id -> one-hot (69 columnas) con las categorías de train.
- geo_level_2_id y geo_level_3_id -> target encoding suavizado (m = 15). En train se
  usa la versión out-of-fold (5 pliegues estratificados, semilla 42) para que ninguna
  fila se codifique con su propia etiqueta; validación y test usan los estadísticos de
  todo train. Ambos se estandarizan con la media y desviación del TE OOF de train.
- 22 binarias sin transformar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config

# Orden de las numéricas escaladas dentro de las 99 features (el del metadata original)
ORDEN_NUMERICAS_ESCALADAS = ["age", "area_percentage", "height_percentage", "count_floors_pre_eq",
                             "count_families"]
COLUMNA_BANDERA_AGE = "age_valor_especial_995"


# ----------------------------------------------------------------- split
def mascara_validacion(datos: pd.DataFrame, semilla: int = config.SEMILLA) -> np.ndarray:
    """Máscara booleana de las filas de validación (20 %, estratificado).

    Es el mismo `train_test_split` con el que el notebook reproduce exactamente
    la columna `split` del parquet original.
    """
    _, val = train_test_split(datos, test_size=config.VAL_SIZE, random_state=semilla,
                              stratify=datos[config.OBJETIVO])
    return datos[config.ID].isin(val[config.ID]).to_numpy()


# --------------------------------------------------------- target encoding
def te_estadisticos(geo: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Suma y conteo de la etiqueta por categoría."""
    return pd.DataFrame({"g": geo.to_numpy(), "y": y.to_numpy()}).groupby("g").y.agg(["sum", "count"])


def te_aplicar(geo: pd.Series, stats: pd.DataFrame, prior: float, m: float = config.SMOOTHING_M) -> pd.Series:
    """Media suavizada (sum + m * prior) / (n + m); categoría no vista -> prior."""
    s = geo.map(stats["sum"])
    n = geo.map(stats["count"])
    return ((s + m * prior) / (n + m)).fillna(prior)


def te_oof(geo_tr: pd.Series, y_tr: pd.Series, prior: float, n_splits: int = config.N_FOLDS_OOF,
           semilla: int = config.SEMILLA) -> np.ndarray:
    """Target encoding out-of-fold: cada fila se codifica sin su pliegue (sin su etiqueta)."""
    geo_tr = geo_tr.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    oof = np.empty(len(geo_tr))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    for idx_fit, idx_enc in skf.split(geo_tr, y_tr):
        st = te_estadisticos(geo_tr.iloc[idx_fit], y_tr.iloc[idx_fit])
        oof[idx_enc] = te_aplicar(geo_tr.iloc[idx_enc], st, prior).to_numpy()
    return oof


# ---------------------------------------------------------- preprocesador
class PreprocesadorRichter:
    """Ajusta los parámetros con train y transforma filas crudas a las 99 features."""

    def __init__(self):
        self.ajustado = False

    # ------------------------------------------------------------- ajuste
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PreprocesadorRichter":
        y = pd.Series(np.asarray(y_train, dtype=float))
        edad = X_train["age"]
        self.mediana_age = float(edad[edad != config.VALOR_ESPECIAL_AGE].median())

        base = self._num_base(X_train)
        self.scaler = {c: (float(base[c].mean()), float(base[c].std(ddof=0))) for c in config.NUMERICAS}

        # categorías ordenadas como texto, igual que en el metadata (geo_level_1_id: '0', '1', '10', ...)
        self.categorias = {c: sorted(X_train[c].astype(str).unique().tolist())
                           for c in config.CAT_BAJA_CARDINALIDAD}

        self.prior = float(y.mean())
        self.te_stats = {}
        self.te_escala_oof = {}
        self.te_escala_congelado = {}
        self._te_oof_train = {}
        for nivel in config.CAT_ALTA_CARDINALIDAD:
            geo = X_train[nivel].reset_index(drop=True)
            st = te_estadisticos(geo, y)
            self.te_stats[nivel] = st
            oof = te_oof(geo, y, self.prior)
            self._te_oof_train[nivel] = oof
            self.te_escala_oof[nivel] = (float(oof.mean()), float(oof.std(ddof=0)))
            # Escala del encoding "congelado" original (incluye la propia etiqueta). Solo se
            # guarda para poder comparar con el parquet histórico; el modelo M5 no la usa.
            congelado = te_aplicar(geo, st, self.prior).to_numpy()
            self.te_escala_congelado[nivel] = (float(congelado.mean()), float(congelado.std(ddof=0)))

        self.feature_cols = self._orden_features()
        self.ajustado = True
        return self

    def fit_transform_oof(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Ajusta y devuelve las features de train con el target encoding out-of-fold."""
        self.fit(X_train, y_train)
        out = self.transform(X_train)
        for nivel in config.CAT_ALTA_CARDINALIDAD:
            mu, sd = self.te_escala_oof[nivel]
            out[f"{nivel}_target_enc_s"] = (self._te_oof_train[nivel] - mu) / sd
        return out

    # ------------------------------------------------------- transformación
    def transform(self, X: pd.DataFrame, escala_te: str = "oof") -> pd.DataFrame:
        """Filas crudas -> DataFrame con las 99 features (sin building_id), en el orden del metadata.

        `escala_te="oof"` es la del modelo M5. `"congelado"` reproduce el parquet histórico
        y solo se usa en las pruebas de trazabilidad.
        """
        if not self.ajustado:
            raise RuntimeError("El preprocesador no está ajustado: llama a fit() o cárgalo con cargar().")
        if escala_te not in ("oof", "congelado"):
            raise ValueError("escala_te debe ser 'oof' o 'congelado'.")
        cols = {}
        for c, cats in self.categorias.items():
            valores = X[c].astype(str).to_numpy()
            for k in cats:
                cols[f"{c}_{k}"] = (valores == k).astype(np.float64)
        base = self._num_base(X)
        for c in config.NUMERICAS:
            mu, sd = self.scaler[c]
            cols[f"{c}_s"] = (base[c].to_numpy() - mu) / sd
        escalas = self.te_escala_oof if escala_te == "oof" else self.te_escala_congelado
        for nivel in config.CAT_ALTA_CARDINALIDAD:
            te = te_aplicar(X[nivel].reset_index(drop=True), self.te_stats[nivel], self.prior).to_numpy()
            mu, sd = escalas[nivel]
            cols[f"{nivel}_target_enc_s"] = (te - mu) / sd
        for c in config.BINARIAS:
            cols[c] = X[c].to_numpy(dtype=np.float64)
        cols[COLUMNA_BANDERA_AGE] = (X["age"].to_numpy() == config.VALOR_ESPECIAL_AGE).astype(np.float64)
        return pd.DataFrame(cols, index=X.index)[self.feature_cols]

    # ------------------------------------------------------------ auxiliares
    def _num_base(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[config.NUMERICAS].astype(float).copy()
        out["age"] = out["age"].where(X["age"] != config.VALOR_ESPECIAL_AGE, self.mediana_age)
        return out

    def _orden_features(self) -> list[str]:
        onehot = [f"{c}_{k}" for c, cats in self.categorias.items() for k in cats]
        numericas = [f"{c}_s" for c in ORDEN_NUMERICAS_ESCALADAS]
        te = [f"{n}_target_enc_s" for n in config.CAT_ALTA_CARDINALIDAD]
        return onehot + numericas + te + list(config.BINARIAS) + [COLUMNA_BANDERA_AGE]

    # ------------------------------------------------------------ persistencia
    def guardar(self, ruta: Path) -> Path:
        """Guarda los parámetros en JSON (legible y sin dependencias de versión de pickle)."""
        datos = {
            "descripcion": "Parámetros del preprocesamiento del modelo M5 (todos ajustados con el split de train).",
            "valor_especial_age": config.VALOR_ESPECIAL_AGE,
            "mediana_age": self.mediana_age,
            "scaler": {c: {"media": mu, "desv": sd} for c, (mu, sd) in self.scaler.items()},
            "categorias": self.categorias,
            "smoothing_m": config.SMOOTHING_M,
            "prior": self.prior,
            "te_escala_oof": {n: {"media": mu, "desv": sd} for n, (mu, sd) in self.te_escala_oof.items()},
            "te_escala_congelado": {n: {"media": mu, "desv": sd} for n, (mu, sd) in self.te_escala_congelado.items()},
            "te_oof": {"n_splits": config.N_FOLDS_OOF, "shuffle": True, "random_state": config.SEMILLA,
                       "estratificado": True},
            "te_stats": {n: {"categoria": st.index.astype(int).tolist(), "sum": st["sum"].tolist(),
                             "count": st["count"].astype(int).tolist()} for n, st in self.te_stats.items()},
            "feature_cols": self.feature_cols,
        }
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: Path) -> "PreprocesadorRichter":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        p = cls()
        p.mediana_age = datos["mediana_age"]
        p.scaler = {c: (v["media"], v["desv"]) for c, v in datos["scaler"].items()}
        p.categorias = datos["categorias"]
        p.prior = datos["prior"]
        p.te_escala_oof = {n: (v["media"], v["desv"]) for n, v in datos["te_escala_oof"].items()}
        p.te_escala_congelado = {n: (v["media"], v["desv"]) for n, v in datos["te_escala_congelado"].items()}
        p.te_stats = {n: pd.DataFrame({"sum": v["sum"], "count": v["count"]},
                                      index=pd.Index(v["categoria"], name="g"))
                      for n, v in datos["te_stats"].items()}
        p.feature_cols = datos["feature_cols"]
        p.ajustado = True
        return p
