"""Subtab Eliminar: mueve una medicion a la papelera (soft-delete)."""
from __future__ import annotations

import streamlit as st

from .. import db
from ..domain import to_es_date_str


def render():
    st.subheader("Eliminar una medicion")
    st.caption("El registro se mueve a la Papelera; se puede restaurar.")

    df = db.load_active().sort_values("FECHA", ascending=False)
    if df.empty:
        st.info("No hay datos para eliminar.")
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
        "Registro a eliminar", list(label_to_id.keys()),
        index=None, placeholder="Selecciona un registro",
    )

    c1, c2 = st.columns(2)
    with c1:
        eliminar = st.button("Eliminar registro", type="primary", disabled=(seleccion is None))
    with c2:
        deshacer = st.button(
            "Deshacer ultimo borrado", type="secondary",
            disabled=(st.session_state.get("ultimo_borrado") is None),
        )

    if eliminar:
        try:
            sel_id = int(label_to_id[seleccion])
            db.soft_delete(sel_id)
            st.session_state["ultimo_borrado"] = sel_id
            st.toast("Registro movido a la Papelera", icon="🗑️")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo eliminar: {e}")

    if deshacer:
        try:
            db.restore(int(st.session_state["ultimo_borrado"]))
            st.session_state["ultimo_borrado"] = None
            st.toast("Se deshizo el ultimo borrado", icon="↩️")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo deshacer: {e}")
