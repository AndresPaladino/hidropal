"""Migracion one-shot/idempotente: CSV de la branch `data` -> Supabase.

Fuente de verdad: la branch `data` del repo (NO master).
  - datos_pozo.csv           -> registros activos (deleted_at = null)
  - datos_pozo_borrados.csv  -> papelera (deleted_at = ahora)

Es idempotente:
  - activos: upsert por fecha (un activo por dia).
  - papelera: inserta solo filas que aun no esten en la papelera (dedupe por
    fecha+nivel+lluvia+extraccion).

Se puede re-correr en el cutover para sincronizar el delta cargado durante el
desarrollo. No borra nada existente.

Uso:
    python migrate_csv_to_supabase.py            # lee de origin/data via git
    python migrate_csv_to_supabase.py --local    # lee los CSV del working dir
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import subprocess
import sys
import tomllib
from pathlib import Path

import pandas as pd

from hidropal.domain import date_to_iso, ensure_datetime_es, normalize_columns

ROOT = Path(__file__).parent
SECRETS = ROOT / ".streamlit" / "secrets.toml"
DATA_BRANCH = "origin/data"


def _load_secrets() -> dict:
    if not SECRETS.exists():
        sys.exit(f"No existe {SECRETS}. Crealo a partir de secrets.toml.example.")
    with open(SECRETS, "rb") as f:
        sec = tomllib.load(f).get("supabase", {})
    url, key = sec.get("url"), sec.get("service_key") or sec.get("key")
    if not (url and key):
        sys.exit("Faltan supabase.url / supabase.service_key en secrets.toml.")
    return {"url": url, "key": key, "table": sec.get("table", "mediciones")}


def _read_csv(name: str, local: bool) -> pd.DataFrame:
    if local:
        path = ROOT / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    out = subprocess.run(
        ["git", "show", f"{DATA_BRANCH}:{name}"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode != 0:
        print(f"  (sin {name} en {DATA_BRANCH})")
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(out.stdout))


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = normalize_columns(df)
    df["FECHA"] = ensure_datetime_es(df["FECHA"])
    for c in ("NIVEL", "LLUVIA", "EXTRACCION"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["LLUVIA"] = df["LLUVIA"].fillna(0)
    df["EXTRACCION"] = df["EXTRACCION"].fillna(0)
    return df.dropna(subset=["FECHA", "NIVEL"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="leer CSV del working dir")
    args = ap.parse_args()

    cfg = _load_secrets()
    from supabase import create_client

    sb = create_client(cfg["url"], cfg["key"]).table(cfg["table"])

    if not args.local:
        subprocess.run(["git", "fetch", "origin", "data"], cwd=ROOT,
                       capture_output=True)

    activos = _prep(_read_csv("datos_pozo.csv", args.local))
    papelera = _prep(_read_csv("datos_pozo_borrados.csv", args.local))
    print(f"Leidos: {len(activos)} activos, {len(papelera)} en papelera.")

    # --- Activos: upsert por fecha ---
    existentes = sb.select("id, fecha").is_("deleted_at", "null").execute().data or []
    by_fecha = {r["fecha"]: r["id"] for r in existentes}
    ins = upd = 0
    for _, r in activos.iterrows():
        fecha = date_to_iso(r["FECHA"])
        vals = {"nivel": float(r["NIVEL"]), "lluvia": float(r["LLUVIA"]),
                "extraccion": float(r["EXTRACCION"])}
        if fecha in by_fecha:
            sb.update(vals).eq("id", by_fecha[fecha]).execute()
            upd += 1
        else:
            sb.insert({"fecha": fecha, "deleted_at": None, **vals}).execute()
            ins += 1
    print(f"Activos -> insertados: {ins}, actualizados: {upd}")

    # --- Papelera: insertar solo lo que falte (dedupe) ---
    trash_db = sb.select("fecha, nivel, lluvia, extraccion").not_.is_(
        "deleted_at", "null").execute().data or []
    seen = {(r["fecha"], float(r["nivel"]), float(r["lluvia"]), float(r["extraccion"]))
            for r in trash_db}
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    tins = 0
    for _, r in papelera.iterrows():
        key = (date_to_iso(r["FECHA"]), float(r["NIVEL"]), float(r["LLUVIA"]),
               float(r["EXTRACCION"]))
        if key in seen:
            continue
        sb.insert({"fecha": key[0], "nivel": key[1], "lluvia": key[2],
                   "extraccion": key[3], "deleted_at": now}).execute()
        seen.add(key)
        tins += 1
    print(f"Papelera -> insertados: {tins}")
    print("Migracion completa.")


if __name__ == "__main__":
    main()
