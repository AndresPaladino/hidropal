"""Seccion Registros: ver, editar y eliminar mediciones, y restaurar la papelera.

Unifica Modificar + Eliminar + Papelera. La seleccion es por tabla
(st.dataframe con seleccion de fila): sin teclado, scrolleable y escala a
cientos de filas.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from .. import db, styles
from ..domain import apply_nivel_offset, validate_input_data

_COLS = ["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]
_COLCFG = {
    "FECHA": st.column_config.DateColumn("Fecha", format="DD/MM/YY"),
    "NIVEL": st.column_config.NumberColumn("Nivel (m)", format="%.2f"),
    "LLUVIA": st.column_config.NumberColumn("Lluvia (mm)", format="%.0f"),
    "EXTRACCION": st.column_config.NumberColumn("Extrac. (lts)", format="%.0f"),
}


def _selected_index(event) -> int | None:
    rows = event.selection.rows if event and event.selection else []
    return rows[0] if rows else None


def render():
    st.subheader("Registros")

    df = db.load_active().sort_values("FECHA", ascending=False).reset_index(drop=True)
    if df.empty:
        st.info("Todavia no hay datos cargados.")
    else:
        st.caption("Tocá un registro para editarlo o eliminarlo.")
        event = st.dataframe(
            df[_COLS], hide_index=True, width="stretch", height=320,
            column_config=_COLCFG, on_select="rerun",
            selection_mode="single-row", key="reg_table",
        )
        idx = _selected_index(event)
        if idx is not None:
            _edit_card(df.iloc[idx])

    if st.session_state.get("ultimo_borrado") is not None:
        if st.button("Deshacer ultimo borrado", width="stretch"):
            try:
                db.restore(int(st.session_state["ultimo_borrado"]))
                st.session_state["ultimo_borrado"] = None
                st.toast("Se deshizo el ultimo borrado", icon="↩️")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo deshacer: {e}")

    st.divider()
    _papelera()


def _edit_card(r):
    sel_id = int(r["id"])
    nivel_original = float(r["NIVEL"])
    fecha = r["FECHA"].date()

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
            "Volumen extraido (lts)",
            value=None if r["EXTRACCION"] == 0 else float(r["EXTRACCION"]), format="%.2f",
        )

    c1, c2 = st.columns(2)
    with c1:
        guardar = st.button("Guardar cambios", type="primary", width="stretch")
    with c2:
        eliminar = st.button("Eliminar", width="stretch", key="btn_eliminar")

    if guardar:
        cleaned, errors = validate_input_data(fecha, nuevo_nivel, nueva_lluvia, nueva_extraccion)
        if errors:
            st.error(" - ".join(errors))
            return
        nivel_input = float(cleaned["NIVEL"]) if cleaned["NIVEL"] is not None else nivel_original
        cambio = not np.isclose(nivel_input, nivel_original, atol=1e-6)
        nivel_final = apply_nivel_offset(nivel_input) if cambio else nivel_original
        try:
            db.update_values(sel_id, nivel_final, cleaned["LLUVIA"], cleaned["EXTRACCION"])
            st.toast("Registro modificado", icon="✅")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo guardar: {e}")

    if eliminar:
        try:
            db.soft_delete(sel_id)
            st.session_state["ultimo_borrado"] = sel_id
            st.toast("Registro movido a la Papelera", icon="🗑️")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo eliminar: {e}")


def _papelera():
    with st.expander("Papelera (restaurar)"):
        trash = db.load_trash().reset_index(drop=True)
        if trash.empty:
            st.caption("La Papelera esta vacia.")
            return
        event = st.dataframe(
            trash[_COLS], hide_index=True, width="stretch", height=220,
            column_config=_COLCFG, on_select="rerun",
            selection_mode="single-row", key="trash_table",
        )
        idx = _selected_index(event)
        if st.button(
            "Restaurar seleccionado", type="primary", width="stretch",
            disabled=(idx is None),
        ):
            try:
                db.restore(int(trash.iloc[idx]["id"]))
                st.toast("Registro restaurado", icon="♻️")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo restaurar: {e}")

        st.caption("")
        if st.button("Vaciar Papelera (definitivo)", width="stretch", key="btn_purge"):
            try:
                db.purge_trash()
                st.toast("Papelera vaciada", icon="🧹")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo vaciar: {e}")
