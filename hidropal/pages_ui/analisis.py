"""Tab Analisis: tabla + las graficas (vista publica de solo-lectura)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from .. import charts, db


def _plot(fig):
    st.pyplot(fig, transparent=True)
    plt.close(fig)  # ponytail: libera la figura, evita fuga de memoria por rerun


def render():
    df = db.load_active_with_derived()

    if df.empty:
        st.info("Todavia no hay datos cargados.")
        return

    # --- Selector de rango ---
    _RANGOS = {"30 dias": 30, "90 dias": 90, "Todo": None}
    rango = st.segmented_control(
        "Rango", list(_RANGOS.keys()), default="Todo",
        label_visibility="collapsed", key="rango_analisis",
    )
    dias = _RANGOS.get(rango or "Todo", None)
    dfv = charts.filtrar_rango(df, dias)

    # --- Graficas ---
    st.subheader("Serie temporal")
    _plot(charts.fig_serie_temporal(dfv))

    st.subheader("Dashboard")
    _plot(charts.fig_dashboard(dfv))

    st.subheader("Comparación de tendencias")
    seleccion = st.pills(
        "Variables a comparar", charts.opciones_comparar(),
        selection_mode="multi",
        default=["Nivel", "Lluvia", "Extracción"],
    )
    if seleccion:
        _plot(charts.fig_comparacion(dfv, seleccion))

    st.subheader("Variación del nivel vs Lluvia acumulada (7d)")
    _plot(charts.fig_scatter_var_lluvia(dfv))

    st.subheader("Variación del nivel vs Volumen extraído")
    _plot(charts.fig_scatter_var_extraccion(dfv))

    st.subheader("Variación del nivel en función de extracción y lluvia")
    _plot(charts.fig_scatter_2d(dfv))

    # --- Tabla completa ---
    with st.expander("Ver tabla de datos"):
        tabla = df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]].sort_values(
            "FECHA", ascending=False
        )
        st.dataframe(
            tabla, hide_index=True, width="stretch",
            column_config={
                "FECHA": st.column_config.DateColumn("FECHA", format="DD/MM/YY"),
                "NIVEL": st.column_config.NumberColumn("NIVEL", format="%.2f"),
                "LLUVIA": st.column_config.NumberColumn("LLUVIA", format="%.0f"),
                "EXTRACCION": st.column_config.NumberColumn("EXTRACCION", format="%.0f"),
            },
        )
