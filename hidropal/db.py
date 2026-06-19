"""Capa de acceso a datos sobre Supabase (Postgres).

Tabla `mediciones` con soft-delete: la papelera son las filas con
`deleted_at` no nulo. Un registro por `fecha` (UNIQUE) -> upsert por dia.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import streamlit as st

from .config import supabase_cfg
from .domain import add_derived, date_to_iso

_DB_COLS = ["id", "fecha", "nivel", "lluvia", "extraccion", "deleted_at"]


# -------------------------
# Cliente
# -------------------------
@st.cache_resource(show_spinner=False)
def _client():
    """Cliente Supabase cacheado. Lanza RuntimeError si falta config."""
    from supabase import create_client

    cfg = supabase_cfg()
    if not cfg:
        raise RuntimeError("Falta configuracion de Supabase en st.secrets")
    return create_client(cfg["url"], cfg["key"])


def _table():
    return _client().table(supabase_cfg()["table"])


def ping() -> tuple[bool, str]:
    """Prueba rapida de conexion."""
    try:
        _table().select("id", count="exact").limit(1).execute()
        return True, "OK"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# -------------------------
# Lectura
# -------------------------
def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_DB_COLS)
    if df.empty:
        df = df.assign(FECHA=pd.Series(dtype="datetime64[ns]"))
        return df.rename(
            columns={"nivel": "NIVEL", "lluvia": "LLUVIA", "extraccion": "EXTRACCION"}
        )[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION", "id"]]
    df = df.rename(
        columns={
            "fecha": "FECHA",
            "nivel": "NIVEL",
            "lluvia": "LLUVIA",
            "extraccion": "EXTRACCION",
        }
    )
    # Supabase devuelve la fecha en ISO (YYYY-MM-DD). NO usar dayfirst aca:
    # con dayfirst=True pandas invierte dia/mes cuando ambos son <= 12.
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%Y-%m-%d", errors="coerce")
    for c in ("NIVEL", "LLUVIA", "EXTRACCION"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["LLUVIA"] = df["LLUVIA"].fillna(0)
    return df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION", "id"]]


def load_active() -> pd.DataFrame:
    """Mediciones activas (papelera excluida), ordenadas por FECHA asc."""
    res = (
        _table()
        .select("*")
        .is_("deleted_at", "null")
        .order("fecha")
        .execute()
    )
    return _rows_to_df(res.data or [])


def load_active_with_derived() -> pd.DataFrame:
    """Activas + columnas derivadas para analisis."""
    df = load_active()
    if df.empty:
        return df
    return add_derived(df)


def load_trash() -> pd.DataFrame:
    """Registros en papelera (deleted_at no nulo)."""
    res = (
        _table()
        .select("*")
        .not_.is_("deleted_at", "null")
        .order("deleted_at", desc=True)
        .execute()
    )
    return _rows_to_df(res.data or [])


# -------------------------
# Escritura
# -------------------------
def upsert_medicion(fecha, nivel: float, lluvia: float, extraccion: float) -> None:
    """Inserta o reemplaza la medicion ACTIVA de un dia.

    Upsert manual (el indice unico es parcial sobre activos, asi que no se usa
    on_conflict de PostgREST): si ya existe un activo para esa fecha, se updatea;
    si no, se inserta.
    """
    fecha_iso = date_to_iso(fecha)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    existing = (
        _table().select("id").is_("deleted_at", "null").eq("fecha", fecha_iso).execute()
    )
    values = {
        "nivel": float(nivel),
        "lluvia": float(lluvia or 0),
        "extraccion": float(extraccion or 0),
        "updated_at": now,
    }
    if existing.data:
        _table().update(values).eq("id", existing.data[0]["id"]).execute()
    else:
        _table().insert({"fecha": fecha_iso, "deleted_at": None, **values}).execute()


def update_values(row_id: int, nivel: float, lluvia: float, extraccion: float) -> None:
    """Actualiza valores de un registro existente por id (fecha fija)."""
    _table().update(
        {
            "nivel": float(nivel),
            "lluvia": float(lluvia or 0),
            "extraccion": float(extraccion or 0),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    ).eq("id", int(row_id)).execute()


def soft_delete(row_id: int) -> None:
    """Mueve un registro a la papelera (marca deleted_at)."""
    _table().update(
        {"deleted_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    ).eq("id", int(row_id)).execute()


def restore(row_id: int) -> None:
    """Restaura un registro de la papelera (limpia deleted_at)."""
    _table().update({"deleted_at": None}).eq("id", int(row_id)).execute()


def purge_trash() -> None:
    """Borra definitivamente todo lo que esta en la papelera."""
    _table().delete().not_.is_("deleted_at", "null").execute()


# -------------------------
# Export / backup CSV
# -------------------------
def export_csv(include_trash: bool = False) -> bytes:
    """Genera el CSV (formato historico) desde la DB."""
    from .domain import to_es_date_str

    df = load_trash() if include_trash else load_active()
    out = df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]].copy()
    out["FECHA"] = to_es_date_str(out["FECHA"])
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# -------------------------
# Backup CSV a GitHub (opcional, manual / baja frecuencia)
# -------------------------
def backup_to_github() -> tuple[bool, str]:
    """Snapshot del CSV de activos al repo (rama de respaldo). Reusa Contents API."""
    import base64

    import requests

    from .config import github_cfg

    cfg = github_cfg()
    if not cfg:
        return False, "Backup a GitHub no configurado (st.secrets['github'])."

    owner, name = cfg["repo"].split("/")
    url = f"https://api.github.com/repos/{owner}/{name}/contents/{cfg['data_path']}"
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "hidropal-app",
    }
    try:
        # SHA actual (si el archivo ya existe) para update.
        sha = None
        r = requests.get(url, headers=headers, params={"ref": cfg["branch"]}, timeout=20)
        if r.status_code == 200:
            sha = r.json().get("sha")
        content = export_csv()
        payload = {
            "message": "chore(backup): snapshot CSV desde Supabase",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": cfg["branch"],
        }
        if cfg.get("author"):
            payload["author"] = cfg["author"]
        if sha:
            payload["sha"] = sha
        r = requests.put(url, headers=headers, json=payload, timeout=20)
        if r.status_code in (200, 201):
            return True, "Respaldo subido a GitHub."
        return False, f"GitHub HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
