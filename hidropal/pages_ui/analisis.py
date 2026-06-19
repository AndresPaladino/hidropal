"""Tab Analisis: tabla + las graficas (vista publica de solo-lectura)."""
from __future__ import annotations

import streamlit as st

from .. import charts, db, insights, styles
from ..domain import to_es_date_str


def render():
    df = db.load_active_with_derived()

    if df.empty:
        st.info("Todavia no hay datos cargados.")
        return

    # --- KPIs de un vistazo ---
    k = insights.kpis(df)
    var = k["variacion_7d"]
    if var is None:
        delta = None
    else:
        arrow = "↑" if var > 0 else ("↓" if var < 0 else "→")
        delta = f"{arrow} {abs(var):.2f} m en 7 dias"
    styles.metric_cards([
        {"icon": "💧", "label": "Nivel actual", "value": f"{k['nivel_actual']:.2f}",
         "unit": "m", "delta": delta},
        {"icon": "🌧️", "label": "Lluvia del mes", "value": f"{k['lluvia_mes']:.0f}",
         "unit": "mm"},
        {"icon": "🚰", "label": "Extraccion del mes", "value": f"{k['extraccion_mes']:.0f}",
         "unit": "lts"},
    ])
    st.caption(
        f"Ultimo registro: {to_es_date_str(df['FECHA']).iloc[-1]}  ·  "
        f"{k['registros']} mediciones"
    )

    st.divider()

    # --- Selector de rango ---
    _RANGOS = {"30 dias": 30, "90 dias": 90, "Todo": None}
    rango = st.segmented_control(
        "Rango", list(_RANGOS.keys()), default="90 dias",
        label_visibility="collapsed", key="rango_analisis",
    )
    dias = _RANGOS.get(rango or "90 dias", 90)
    dfv = charts.filtrar_rango(df, dias)

    # --- Graficas principales ---
    st.subheader("Serie temporal")
    st.pyplot(charts.fig_serie_temporal(dfv))

    # --- Graficas avanzadas (plegadas) ---
    with st.expander("Mas analisis"):
        st.subheader("Dashboard")
        st.pyplot(charts.fig_dashboard(dfv))

        st.subheader("Comparación de tendencias")
        seleccion = st.multiselect(
            "Variables a comparar", charts.opciones_comparar(),
            placeholder="Selecciona una o más variables",
            default=["Nivel", "Lluvia", "Extracción"],
        )
        if seleccion:
            st.pyplot(charts.fig_comparacion(dfv, seleccion))

        st.subheader("Variación del nivel vs Lluvia acumulada (7d)")
        st.pyplot(charts.fig_scatter_var_lluvia(dfv))

        st.subheader("Variación del nivel vs Volumen extraído")
        st.pyplot(charts.fig_scatter_var_extraccion(dfv))

        st.subheader("Variación del nivel en función de extracción y lluvia")
        st.pyplot(charts.fig_scatter_2d(dfv))

    # --- Tabla completa ---
    with st.expander("Ver tabla de datos"):
        tabla = df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]].sort_values(
            "FECHA", ascending=False
        )
        st.dataframe(
            tabla, hide_index=True, use_container_width=True,
            column_config={"FECHA": st.column_config.DateColumn("FECHA", format="DD/MM/YY")},
        )
