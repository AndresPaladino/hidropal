"""Subtab Restaurar: recupera o vacia la Papelera."""
from __future__ import annotations

import streamlit as st

from .. import db
from ..domain import to_es_date_str


def render():
    st.subheader("Papelera")

    trash = db.load_trash()
    if trash.empty:
        st.info("La Papelera esta vacia.")
        return

    trash = trash.copy()
    trash["LABEL"] = (
        to_es_date_str(trash["FECHA"])
        + "  |  Nivel=" + trash["NIVEL"].map(lambda x: f"{x:.2f}m")
        + "  Lluvia=" + trash["LLUVIA"].map(lambda x: f"{x:.0f}mm")
        + "  Extr=" + trash["EXTRACCION"].map(lambda x: f"{x:.0f}lts")
    )
    label_to_id = dict(zip(trash["LABEL"], trash["id"]))

    seleccion = st.selectbox(
        "Registro a restaurar", list(label_to_id.keys()),
        index=None, placeholder="Selecciona un registro",
    )

    if st.button("Restaurar", type="primary", disabled=(seleccion is None)):
        try:
            db.restore(int(label_to_id[seleccion]))
            st.toast("Registro restaurado", icon="♻️")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo restaurar: {e}")

    st.divider()
    with st.expander("Vaciar Papelera (borra definitivamente)"):
        st.warning("Esto elimina para siempre todos los registros de la Papelera.")
        if st.button("Vaciar Papelera"):
            try:
                db.purge_trash()
                st.toast("Papelera vaciada", icon="🧹")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo vaciar: {e}")
