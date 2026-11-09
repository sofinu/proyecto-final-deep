"""Extrae las tablas de resultados desde las salidas guardadas del notebook.

Las tablas que el notebook escribió en Colab (T01-T18, escalera) no se descargaron. Sus valores
siguen guardados como salidas HTML dentro del .ipynb; este script los recupera sin reejecutar
nada, para que reports/tables/ contenga exactamente las cifras reportadas.

Uso:  python scripts/extraer_tablas_notebook.py
"""
from __future__ import annotations

import csv
import json
from html.parser import HTMLParser
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "proyecto_integrador_richter.ipynb"
DESTINO = RAIZ / "reports" / "tables"

# (texto que identifica la celda, posición de la salida HTML dentro de la celda, archivo)
TABLAS = [
    ('guardar_tabla(auditoria, "T01_auditoria_archivos")', 0, "T01_auditoria_archivos"),
    ('"T02_diccionario_variables"', 1, "T02_diccionario_variables"),
    ('"T03_distribucion_clases"', 0, "T03_distribucion_clases"),
    ('"T04_duplicados"', 0, "T04_duplicados"),
    ('"T05_resumen_numericas"', 1, "T05_resumen_numericas"),
    ('"T06_split_train_val"', 0, "T06_split_train_val"),
    ('"T07_verificacion_escalado"', 0, "T07_verificacion_escalado"),
    ('"T08_reconstruccion_target_encoding"', 0, "T08_reconstruccion_target_encoding"),
    ('"T09_fuga_por_tamano_de_categoria"', 0, "T09_fuga_correlacion_por_tamano_de_categoria"),
    ('"T10_impacto_fuga_congelado_vs_oof"', 0, "T10_impacto_fuga_congelado_vs_oof"),
    ('"T11_tabla_transformaciones"', 0, "T11_tabla_transformaciones"),
    ('"T12_test_compatibilidad"', 0, "T12_test_compatibilidad"),
    ('"T13_test_deriva"', 1, "T13_test_deriva"),
    ('"T14_edad_por_clase"', 0, "T14_edad_por_clase"),
    ('"T15_cimentacion_por_clase"', 0, "T15_cimentacion_por_clase"),
    ('"T16_geo1_por_clase"', 0, "T16_geo1_por_clase"),
    ('alphas = [1e-5, 1e-4, 1e-3, 1e-2]', 0, "M1_grid_alpha_val_A"),
    ('tabla2_escalera_optimizacion.csv', 0, "tabla2_escalera_optimizacion"),
    ('# 3. Tabla resumen por paso', 0, "errores_por_clase_escalera_val_B"),
    ('# Referencia: LightGBM', 0, "M5_vs_LightGBM_val_B"),
    ('"T17_modelo_final_seleccionado"', 0, "T17_modelo_final_seleccionado"),
    ('"T18_distribucion_clases_test_vs_train"', 0, "T18_distribucion_clases_test_vs_train"),
]


class _Tabla(HTMLParser):
    """Lee la primera <table> generada por pandas (encabezados de varias filas incluidos)."""

    def __init__(self):
        super().__init__()
        self.filas, self._fila, self._celda, self._dentro = [], None, None, False
        self._pendientes = {}   # columna -> [texto, filas restantes] (celdas con rowspan)

    def _rellenar_rowspan(self):
        while len(self._fila) in self._pendientes:
            col = len(self._fila)
            texto, restantes = self._pendientes[col]
            self._fila.append(texto)
            if restantes <= 1:
                del self._pendientes[col]
            else:
                self._pendientes[col][1] = restantes - 1

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._dentro = True
        elif self._dentro and tag == "tr":
            self._fila = []
        elif self._dentro and tag in ("td", "th"):
            self._rellenar_rowspan()
            self._celda = []
            a = dict(attrs)
            self._colspan, self._rowspan = int(a.get("colspan", 1)), int(a.get("rowspan", 1))

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._celda is not None:
            texto = " ".join("".join(self._celda).split())
            if self._rowspan > 1:
                self._pendientes[len(self._fila)] = [texto, self._rowspan - 1]
            self._fila.extend([texto] * self._colspan)
            self._celda = None
        elif tag == "tr" and self._fila is not None:
            self.filas.append(self._fila)
            self._fila = None
        elif tag == "table":
            self._dentro = False

    def handle_data(self, data):
        if self._celda is not None:
            self._celda.append(data)


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    DESTINO.mkdir(parents=True, exist_ok=True)
    for marca, posicion, nombre in TABLAS:
        celda = next(c for c in nb["cells"] if c["cell_type"] == "code" and marca in "".join(c["source"]))
        htmls = ["".join(o["data"]["text/html"]) for o in celda.get("outputs", [])
                 if "text/html" in o.get("data", {}) and "<table" in "".join(o["data"]["text/html"])]
        parser = _Tabla()
        parser.feed(htmls[posicion])
        ancho = max(len(f) for f in parser.filas)
        with open(DESTINO / f"{nombre}.csv", "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(fila + [""] * (ancho - len(fila)) for fila in parser.filas)
        print(f"{nombre}.csv  ({len(parser.filas)} filas)")


if __name__ == "__main__":
    main()
