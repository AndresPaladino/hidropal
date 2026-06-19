"""Subtab Modificar: editar la medicion de un dia (elegido por fecha)."""
from __future__ import annotations

from datetime import date

import numpy as np
import streamlit as st

from .. import db, styles
from ..domain import apply_nivel_offset, validate_input_data


def render():
    st.subheader("Modificar una medicion")

    df = db.load_active()
    if df.empty:
        st.info("No hay datos para modificar.")
        return

    fecha = st.date_input(
        "¿Que dia queres modificar?", value=date.today(), max_value=date.today(),
        format="DD/MM/YYYY", key="modificar_fecha",
    )
    sel = df[df["FECHA"].dt.date == fecha] if fecha else df.iloc[0:0]
    if sel.empty:
        st.info(f"No hay registro del {fecha.strftime('%d/%m/%Y')}. Elegi otro dia.")
        return

    r = sel.iloc[0]
    sel_id = int(r["id"])
    nivel_original = float(r["NIVEL"])

    styles.metric_cards([
        {"icon": "💧", "label": "Nivel", "value": f"{nivel_original:.2f}", "unit": "m"},
        {"icon": "🌧️", "label": "Lluvia", "value": f"{float(r['LLUVIA']):.0f}", "unit": "mm"},
        {"icon": "🚰", "label": "Extraccion", "value": f"{float(r['EXTRACCION']):.0f}", "unit": "lts"},
    ])

    with st.container(border=True):
        nuevo_nivel = st.number_input("Nivel del agua (m)", value=nivel_original, format="%.2f")
        st.caption("Si cambias el nivel, se aplica el ajuste de la cinta (-0.17 m).")
        nueva_lluvia = st.number_input(
            "Lluvia caida (mm)", value=None if r["LLUVIA"] == 0 else float(r["LLUVIA"]),
            format="%.2f",
        )
        nueva_extraccion = st.number_input(
            "Volumen extraido (lts)", value=None if r["EXTRACCION"] == 0 else float(r["EXTRACCION"]),
            format="%.2f",
        )

    if st.button("Guardar cambios", type="primary", width="stretch"):
        cleaned, errors = validate_input_data(fecha, nuevo_nivel, nueva_lluvia, nueva_extraccion)
        if errors:
            st.error(" - ".join(errors))
            return
        nivel_input = float(cleaned["NIVEL"]) if cleaned["NIVEL"] is not None else nivel_original
        cambio_nivel = not np.isclose(nivel_input, nivel_original, atol=1e-6)
        nivel_final = apply_nivel_offset(nivel_input) if cambio_nivel else nivel_original
        try:
            db.update_values(sel_id, nivel_final, cleaned["LLUVIA"], cleaned["EXTRACCION"])
            st.toast("Registro modificado", icon="✅")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo guardar: {e}")
