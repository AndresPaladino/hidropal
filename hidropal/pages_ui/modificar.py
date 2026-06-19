"""Subtab Modificar: editar una medicion existente (fecha fija)."""
from __future__ import annotations

import numpy as np
import streamlit as st

from .. import db
from ..domain import apply_nivel_offset, to_es_date_str, validate_input_data


def render():
    st.subheader("Modificar una medicion")

    df = db.load_active().sort_values("FECHA", ascending=False)
    if df.empty:
        st.info("No hay datos para modificar.")
        return

    df = df.copy()
    df["LABEL"] = (
        to_es_date_str(df["FECHA"])
        + "  |  Nivel=" + df["NIVEL"].map(lambda x: f"{x:.2f}m")
        + "  Lluvia=" + df["LLUVIA"].map(lambda x: f"{x:.0f}mm")
        + "  Extr=" + df["EXTRACCION"].map(lambda x: f"{x:.0f}lts")
    )
    label_to_id = dict(zip(df["LABEL"], df["id"]))

    seleccion = st.selectbox(
        "Registro a modificar", list(label_to_id.keys()),
        index=None, placeholder="Selecciona un registro",
    )
    if not seleccion:
        return

    sel_id = int(label_to_id[seleccion])
    r = df[df["id"] == sel_id].iloc[0]
    nivel_original = float(r["NIVEL"])

    st.info(f"Modificando el {to_es_date_str(df['FECHA'][df['id'] == sel_id]).iloc[0]}")

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

    if st.button("Guardar cambios", type="primary"):
        fecha_actual = r["FECHA"].date()
        cleaned, errors = validate_input_data(
            fecha_actual, nuevo_nivel, nueva_lluvia, nueva_extraccion
        )
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
