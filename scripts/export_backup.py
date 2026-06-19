"""Export de respaldo: Supabase -> CSV (standalone, sin Streamlit).

Pensado para correr en GitHub Actions (cron diario). La consulta a Supabase
ademas cuenta como actividad y mantiene el proyecto free despierto (keep-alive).

Genera, en el directorio actual:
  - datos_pozo.csv           (registros activos)
  - datos_pozo_borrados.csv  (papelera)

Formato historico: FECHA (dd/mm/aaaa), NIVEL, LLUVIA, EXTRACCION.

Variables de entorno requeridas:
  SUPABASE_URL, SUPABASE_KEY   (service key; ver Settings -> API en Supabase)
  SUPABASE_TABLE               (opcional, default 'mediciones')
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from supabase import create_client

_COLS = ["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]


def _fetch(table, *, trash: bool) -> pd.DataFrame:
    q = table.select("fecha, nivel, lluvia, extraccion").order("fecha")
    q = q.not_.is_("deleted_at", "null") if trash else q.is_("deleted_at", "null")
    rows = q.execute().data or []
    df = pd.DataFrame(rows, columns=["fecha", "nivel", "lluvia", "extraccion"])
    if df.empty:
        return pd.DataFrame(columns=_COLS)
    df["FECHA"] = pd.to_datetime(df["fecha"], errors="coerce").dt.strftime("%d/%m/%Y")
    return df.rename(
        columns={"nivel": "NIVEL", "lluvia": "LLUVIA", "extraccion": "EXTRACCION"}
    )[_COLS]


def main() -> int:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    table_name = os.environ.get("SUPABASE_TABLE", "mediciones")
    if not (url and key):
        print("ERROR: faltan SUPABASE_URL / SUPABASE_KEY", file=sys.stderr)
        return 1

    table = create_client(url, key).table(table_name)
    activos = _fetch(table, trash=False)
    papelera = _fetch(table, trash=True)

    activos.to_csv("datos_pozo.csv", index=False)
    papelera.to_csv("datos_pozo_borrados.csv", index=False)
    print(f"OK: {len(activos)} activos, {len(papelera)} en papelera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
