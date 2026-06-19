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

    # --- Graficas ---
    st.subheader("Serie temporal")
    st.plotly_chart(charts.fig_serie_temporal(df), use_container_width=True)

    st.subheader("Dashboard")
    st.plotly_chart(charts.fig_dashboard(df), use_container_width=True)

    st.subheader("Comparacion de tendencias")
    opciones = list(charts.variables_para_comparar(df).keys())
    # st.pills: seleccion multiple por toque, sin teclado (mobile-friendly).
    seleccion = st.pills(
        "Variables a comparar", opciones, selection_mode="multi",
        default=["Nivel", "Lluvia", "Extraccion"],
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
