"""Logica de dominio: validacion, fechas, derivadas, offset de nivel."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .config import COLS, DATE_OUT_FMT, NIVEL_OFFSET, RENAME_MAP


# -------------------------
# Normalizacion de columnas
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lleva las columnas a FECHA, NIVEL, LLUVIA, EXTRACCION (mayusculas)."""
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={c: RENAME_MAP.get(c, c) for c in df.columns})
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS]


# -------------------------
# Utilidades de fecha (interpreta dia/mes/anio)
# -------------------------
def ensure_datetime_es(s: pd.Series) -> pd.Series:
    """Convierte a datetime interpretando dia/mes/anio."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def to_es_date_str(s: pd.Series) -> pd.Series:
    """Convierte a string dd/mm/aaaa (sin romper strings ya formateados)."""
    if np.issubdtype(s.dtype, np.datetime64):
        return s.dt.strftime(DATE_OUT_FMT)
    parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
    out = s.astype("object").copy()
    mask = parsed.notna()
    out[mask] = parsed[mask].dt.strftime(DATE_OUT_FMT)
    return out


def date_to_iso(d) -> str:
    """date/datetime/str -> 'YYYY-MM-DD' para Postgres."""
    return ensure_datetime_es(pd.Series([d])).dt.strftime("%Y-%m-%d").iloc[0]


# -------------------------
# Offset de nivel
# -------------------------
def apply_nivel_offset(nivel_medido: float) -> float:
    """Resta el offset de la cinta al nivel medido por el usuario."""
    return float(nivel_medido) - NIVEL_OFFSET


# -------------------------
# Derivadas para analisis
# -------------------------
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega VARIACION_NIVEL y LLUVIA_ACUM_7D (orden por FECHA)."""
    df = df.sort_values("FECHA").copy()
    df["VARIACION_NIVEL"] = (-df["NIVEL"]).diff()
    df["LLUVIA_ACUM_7D"] = df["LLUVIA"].rolling(window=7).sum()
    return df


# -------------------------
# Validacion de entrada
# -------------------------
def validate_input_data(fecha, nivel, lluvia, extraccion):
    """Valida y limpia datos de entrada. Retorna (dict_limpio, lista_errores)."""
    errors: list[str] = []

    if fecha is None:
        fecha = date.today()
    if fecha > date.today():
        errors.append("La fecha no puede ser futura")

    if nivel is None:
        errors.append("El nivel no puede ser vacio")
    elif nivel <= 0:
        errors.append("El nivel debe ser mayor a 0")

    if lluvia is None:
        lluvia = 0.0
    elif lluvia < 0:
        errors.append("La lluvia no puede ser negativa")

    if extraccion is None:
        extraccion = 0.0
    elif extraccion < 0:
        errors.append("La extraccion no puede ser negativa")

    cleaned = {
        "FECHA": fecha,
        "NIVEL": nivel if nivel is not None and nivel > 0 else None,
        "LLUVIA": lluvia,
        "EXTRACCION": extraccion,
    }
    return cleaned, errors
