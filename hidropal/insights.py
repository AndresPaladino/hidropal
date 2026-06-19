"""Resumenes descriptivos (KPIs) sobre los datos. Sin alertas ni umbrales:
solo un vistazo de lo ya cargado.
"""
from __future__ import annotations

import pandas as pd


def kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs de un vistazo a partir de las mediciones activas.

    df: columnas FECHA (datetime), NIVEL, LLUVIA, EXTRACCION, ordenado por FECHA.
    Retorna dict con valores y deltas listos para st.metric.
    """
    d = df.sort_values("FECHA")
    ultimo = d.iloc[-1]
    fecha_ult = ultimo["FECHA"]

    # Variacion de nivel en los ultimos 7 dias (vs el registro ~7 dias antes).
    hace_7d = fecha_ult - pd.Timedelta(days=7)
    previos = d[d["FECHA"] <= hace_7d]
    var_7d = None
    if not previos.empty:
        var_7d = float(ultimo["NIVEL"]) - float(previos.iloc[-1]["NIVEL"])

    # Totales del mes en curso (mes del ultimo registro).
    mes = d[(d["FECHA"].dt.year == fecha_ult.year) & (d["FECHA"].dt.month == fecha_ult.month)]

    return {
        "fecha_ultimo": fecha_ult,
        "nivel_actual": float(ultimo["NIVEL"]),
        "variacion_7d": var_7d,
        "lluvia_mes": float(mes["LLUVIA"].sum()),
        "extraccion_mes": float(mes["EXTRACCION"].sum()),
        "registros": int(len(d)),
    }
