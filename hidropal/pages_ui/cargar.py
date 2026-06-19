"""Subtab Cargar: alta de la medicion del dia (accion diaria principal).

Diseno enfocado: fecha + un campo de nivel destacado con preview en vivo del
ajuste de cinta (-0.17 m). Lluvia/extraccion van plegados (casi siempre 0).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .. import db
from ..domain import apply_nivel_offset, validate_input_data

_KEYS = ("cargar_nivel", "cargar_lluvia", "cargar_extraccion")


def _reset_inputs():
    for k in _KEYS:
        st.session_state.pop(k, None)


def render():
    st.subheader("Cargar medicion del dia")

    df = db.load_active()

    fecha = st.date_input(
        "Fecha", value=date.today(), max_value=date.today(), format="DD/MM/YYYY",
        key="cargar_fecha",
    )

    existing = pd.DataFrame()
    if fecha is not None and not df.empty:
        existing = df[df["FECHA"].dt.date == fecha]

    # Ya hay datos para esa fecha -> mostrar y derivar a Modificar
    if not existing.empty:
        r = existing.iloc[0]
        st.success(f"Ya cargaste el {fecha.strftime('%d/%m/%Y')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nivel", f"{float(r['NIVEL']):.2f} m")
        c2.metric("Lluvia", f"{float(r['LLUVIA']):.0f} mm")
        c3.metric("Extraccion", f"{float(r['EXTRACCION']):.0f} lts")
        st.info("Para cambiarlos usa la pestania **Modificar**, o elegi otra fecha.")
    else:
        with st.container(border=True):
            nivel = st.number_input(
                "Nivel del agua medido (m)", value=None, format="%.2f",
                step=0.01, key="cargar_nivel",
                placeholder="Ej: 1.05",
            )
            # Preview en vivo del ajuste de cinta
            if nivel is not None and nivel > 0:
                st.caption(
                    f"Se guardara con el ajuste de la cinta: **{apply_nivel_offset(nivel):.2f} m**"
                )
            else:
                st.caption("Se le resta 0.17 m (ajuste de la cinta) al guardar.")

            with st.expander("¿Llovio o extrajiste agua? (opcional)"):
                lluvia = st.number_input(
                    "Lluvia caida (mm)", value=None, format="%.2f", step=1.0,
                    key="cargar_lluvia",
                )
                extraccion = st.number_input(
                    "Volumen extraido (lts)", value=None, format="%.2f", step=10.0,
                    key="cargar_extraccion",
                )

        if st.button("Guardar", type="primary", width="stretch"):
            cleaned, errors = validate_input_data(fecha, nivel, lluvia, extraccion)
            if errors:
                st.error(" - ".join(errors))
            else:
                try:
                    db.upsert_medicion(
                        cleaned["FECHA"],
                        apply_nivel_offset(cleaned["NIVEL"]),
                        cleaned["LLUVIA"],
                        cleaned["EXTRACCION"],
                    )
                    _reset_inputs()
                    st.toast("Datos guardados", icon="✅")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"No se pudo guardar: {e}")

    # Ultimos registros
    if not df.empty:
        st.caption("Ultimos 10 registros:")
        head = df.sort_values("FECHA", ascending=False).head(10)
        head = head[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]]
        st.dataframe(
            head, hide_index=True, width="stretch",
            column_config={"FECHA": st.column_config.DateColumn("FECHA", format="DD/MM/YY")},
        )
