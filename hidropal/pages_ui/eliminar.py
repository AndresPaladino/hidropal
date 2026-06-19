"""Subtab Eliminar: mover a la papelera la medicion de un dia (elegido por fecha)."""
from __future__ import annotations

from datetime import date

import streamlit as st

from .. import db, styles


def render():
    st.subheader("Eliminar una medicion")
    st.caption("El registro se mueve a la Papelera; se puede restaurar.")

    df = db.load_active()
    if df.empty:
        st.info("No hay datos para eliminar.")
        return

    fecha = st.date_input(
        "¿Que dia queres eliminar?", value=date.today(), max_value=date.today(),
        format="DD/MM/YYYY", key="eliminar_fecha",
    )
    sel = df[df["FECHA"].dt.date == fecha] if fecha else df.iloc[0:0]

    if sel.empty:
        st.info(f"No hay registro del {fecha.strftime('%d/%m/%Y')}. Elegi otro dia.")
    else:
        r = sel.iloc[0]
        sel_id = int(r["id"])
        styles.metric_cards([
            {"icon": "💧", "label": "Nivel", "value": f"{float(r['NIVEL']):.2f}", "unit": "m"},
            {"icon": "🌧️", "label": "Lluvia", "value": f"{float(r['LLUVIA']):.0f}", "unit": "mm"},
            {"icon": "🚰", "label": "Extraccion", "value": f"{float(r['EXTRACCION']):.0f}", "unit": "lts"},
        ])
        if st.button("Eliminar este registro", type="primary", width="stretch"):
            try:
                db.soft_delete(sel_id)
                st.session_state["ultimo_borrado"] = sel_id
                st.toast("Registro movido a la Papelera", icon="🗑️")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo eliminar: {e}")

    if st.session_state.get("ultimo_borrado") is not None:
        if st.button("Deshacer ultimo borrado", type="secondary", width="stretch"):
            try:
                db.restore(int(st.session_state["ultimo_borrado"]))
                st.session_state["ultimo_borrado"] = None
                st.toast("Se deshizo el ultimo borrado", icon="↩️")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo deshacer: {e}")
