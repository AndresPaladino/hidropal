"""Subtab Cargar: alta de una medicion nueva (con offset de nivel)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .. import db
from ..domain import apply_nivel_offset, to_es_date_str, validate_input_data


def render():
    st.subheader("Cargar nueva medicion")

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
        st.success(f"Ya hay datos guardados del {fecha.strftime('%d/%m/%Y')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nivel", f"{float(r['NIVEL']):.2f} m")
        c2.metric("Lluvia", f"{float(r['LLUVIA']):.0f} mm")
        c3.metric("Extraccion", f"{float(r['EXTRACCION']):.0f} lts")
        st.info("Para cambiarlos, usa la pestania 'Modificar' o elegi otra fecha.")
    else:
        with st.form("form_cargar", clear_on_submit=True):
            nivel = st.number_input("Nivel del agua medido (m)", value=None, format="%.2f")
            lluvia = st.number_input("Lluvia caida (mm)", value=None, format="%.2f")
            extraccion = st.number_input("Volumen extraido (lts)", value=None, format="%.2f")
            st.caption("Al nivel medido se le aplica el ajuste de la cinta (-0.17 m).")
            submitted = st.form_submit_button("Guardar", type="primary")

        if submitted:
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
                    st.toast("Datos guardados", icon="✅")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"No se pudo guardar: {e}")

    # Ultimos registros
    if not df.empty:
        st.caption("Ultimos 10 registros:")
        head = df.sort_values("FECHA", ascending=False).head(10).copy()
        head = head[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]]
        st.dataframe(
            head, hide_index=True, use_container_width=True,
            column_config={"FECHA": st.column_config.DateColumn("FECHA", format="DD/MM/YY")},
        )
