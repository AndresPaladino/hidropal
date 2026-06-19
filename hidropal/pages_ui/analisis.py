"""Tab Analisis: tabla + las graficas (vista publica de solo-lectura)."""
from __future__ import annotations

import streamlit as st

from .. import charts, db
from ..domain import to_es_date_str


def render():
    df = db.load_active_with_derived()

    if df.empty:
        st.info("Todavia no hay datos cargados.")
        return

    # --- Resumen rapido (ultimo registro) ---
    ultimo = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel", f"{ultimo['NIVEL']:.2f} m")
    c2.metric("Lluvia", f"{ultimo['LLUVIA']:.0f} mm")
    c3.metric("Extraccion", f"{ultimo['EXTRACCION']:.0f} lts")
    st.caption(f"Ultimo registro: {to_es_date_str(df['FECHA']).iloc[-1]}")

    st.divider()

    # --- Graficas ---
    st.subheader("Serie temporal")
    st.plotly_chart(charts.fig_serie_temporal(df), use_container_width=True)

    st.subheader("Dashboard")
    st.plotly_chart(charts.fig_dashboard(df), use_container_width=True)

    st.subheader("Comparacion de tendencias")
    opciones = list(charts.variables_para_comparar(df).keys())
    seleccion = st.multiselect(
        "Variables a comparar", opciones,
        default=["Nivel", "Lluvia", "Extraccion"],
        placeholder="Selecciona una o mas variables",
    )
    if seleccion:
        st.plotly_chart(charts.fig_comparacion(df, seleccion), use_container_width=True)

    st.subheader("Variacion del nivel vs Lluvia acumulada (7d)")
    st.plotly_chart(
        charts.fig_scatter_var(df, "LLUVIA_ACUM_7D", "Lluvia acumulada (mm)"),
        use_container_width=True,
    )

    st.subheader("Variacion del nivel vs Volumen extraido")
    st.plotly_chart(
        charts.fig_scatter_var(df, "EXTRACCION", "Extraccion (lts)"),
        use_container_width=True,
    )

    st.subheader("Variacion del nivel segun extraccion y lluvia")
    st.plotly_chart(charts.fig_scatter_2d(df), use_container_width=True)

    # --- Tabla completa ---
    with st.expander("Ver tabla de datos"):
        tabla = df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]].sort_values(
            "FECHA", ascending=False
        )
        st.dataframe(
            tabla, hide_index=True, use_container_width=True,
            column_config={"FECHA": st.column_config.DateColumn("FECHA", format="DD/MM/YY")},
        )
