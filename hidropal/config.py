"""Constantes y acceso a configuracion (st.secrets)."""
from __future__ import annotations

import streamlit as st

# -------------------------
# Columnas y dominio
# -------------------------
COLS = ["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]

# Offset de la cinta de medicion: al nivel medido por el usuario se le resta
# este valor antes de guardarlo. Logica historica preservada del app original.
NIVEL_OFFSET = 0.17

# Formato de fecha de salida (es).
DATE_OUT_FMT = "%d/%m/%Y"

# -------------------------
# Paleta de colores consistente (igual a la version original)
# -------------------------
COLOR_PALETTE = {
    "NIVEL": "#d62728",           # Rojo
    "LLUVIA": "#000fff",          # Azul
    "EXTRACCION": "#2ca02c",      # Verde
    "VARIACION_NIVEL": "#ff7f0e",  # Naranja
    "LLUVIA_ACUM_7D": "#9467bd",  # Purpura
}

# -------------------------
# Normalizacion de columnas (acepta nombres largos o cortos)
# -------------------------
RENAME_MAP = {
    "FECHA": "FECHA",
    "NIVEL DE AGUA (MTS.)": "NIVEL",
    "LLUVIA CAIDA (MM)": "LLUVIA",
    "VOLUMEN EXTRAIDO (LTS.)": "EXTRACCION",
    "NIVEL": "NIVEL",
    "LLUVIA": "LLUVIA",
    "EXTRACCION": "EXTRACCION",
}


# -------------------------
# Secrets: Supabase
# -------------------------
def supabase_cfg() -> dict | None:
    """Lee config de Supabase desde st.secrets. Retorna dict o None si falta."""
    try:
        sec = st.secrets.get("supabase", {})
        url = sec.get("url")
        key = sec.get("service_key") or sec.get("key")
        table = sec.get("table", "mediciones")
        if url and key:
            return {"url": url, "key": key, "table": table}
    except Exception:
        pass
    return None


def supabase_enabled() -> bool:
    return supabase_cfg() is not None


# -------------------------
# Secrets: autenticacion (PIN de edicion)
# -------------------------
def auth_cfg() -> dict | None:
    """Lee config de auth desde st.secrets. Retorna dict o None si falta."""
    try:
        sec = st.secrets.get("app", {})
        pin_hash = sec.get("pin_hash")
        cookie_secret = sec.get("cookie_secret")
        if pin_hash and cookie_secret:
            return {
                "pin_hash": pin_hash,
                "cookie_secret": cookie_secret,
                # ~1 anio en dias
                "cookie_days": int(sec.get("cookie_days", 365)),
                "cookie_name": sec.get("cookie_name", "hidropal_auth"),
            }
    except Exception:
        pass
    return None


# -------------------------
# Secrets: backup a GitHub (opcional)
# -------------------------
def github_cfg() -> dict | None:
    """Lee config de GitHub (solo para backup CSV manual). None si falta."""
    try:
        sec = st.secrets.get("github", {})
        token = sec.get("GITHUB_TOKEN")
        repo = sec.get("GITHUB_REPO")
        if token and repo:
            return {
                "token": token,
                "repo": repo,
                "branch": sec.get("GITHUB_BRANCH", "data"),
                "data_path": sec.get("GITHUB_DATA_PATH", "datos_pozo.csv"),
                "author": _named(sec, "GITHUB_COMMIT_AUTHOR_NAME", "GITHUB_COMMIT_AUTHOR_EMAIL"),
            }
    except Exception:
        pass
    return None


def _named(sec, name_key, email_key):
    name = sec.get(name_key)
    email = sec.get(email_key)
    if name and email:
        return {"name": name, "email": email}
    return None
