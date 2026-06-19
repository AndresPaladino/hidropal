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

    # --- Selector de rango (mobile: evita comprimir años en 360px) ---
    _RANGOS = {"30 dias": 30, "90 dias": 90, "Todo": None}
    rango = st.segmented_control(
        "Rango", list(_RANGOS.keys()), default="90 dias",
        label_visibility="collapsed", key="rango_analisis",
    )
    dias = _RANGOS.get(rango or "90 dias", 90)
    dfv = charts.filtrar_rango(df, dias)

    # --- Graficas principales ---
    st.subheader("Nivel del pozo")
    st.caption("Mas arriba = mas agua (el eje esta invertido).")
    st.plotly_chart(charts.fig_nivel(dfv), use_container_width=True, config=charts.PLOTLY_CONFIG)

    st.subheader("Nivel y lluvia")
    st.caption("¿La lluvia recupera el pozo? Cada uno con su escala real.")
    st.plotly_chart(charts.fig_nivel_lluvia(dfv), use_container_width=True, config=charts.PLOTLY_CONFIG)

    st.subheader("Extraccion")
    st.plotly_chart(charts.fig_extraccion(dfv), use_container_width=True, config=charts.PLOTLY_CONFIG)

    # --- Graficas avanzadas (plegadas) ---
    with st.expander("Mas analisis"):
        st.markdown("**Dashboard completo**")
        st.plotly_chart(charts.fig_dashboard(dfv), use_container_width=True, config=charts.PLOTLY_CONFIG)

        st.markdown("**Comparacion de tendencias**")
        # st.pills: seleccion multiple por toque, sin teclado (mobile-friendly).
        seleccion = st.pills(
            "Variables a comparar", charts.opciones_comparar(), selection_mode="multi",
            default=["Nivel", "Lluvia"],
        )
        # El grafico se adapta: 1 o 2 variables -> escala real; 3+ -> relativo.
        if seleccion:
            n = len(seleccion)
            if n <= 2:
                st.caption(f"Mostrando {n} variable(s) con su escala real.")
            else:
                st.caption(f"{n} variables en escala relativa (0-100%): compara la forma, no la cantidad. Toca un punto para ver el valor real.")
            st.plotly_chart(
                charts.fig_comparacion(dfv, seleccion),
                use_container_width=True, config=charts.PLOTLY_CONFIG,
            )

        st.markdown("**Variacion del nivel vs Lluvia acumulada (7d)**")
        st.plotly_chart(
            charts.fig_scatter_var(dfv, "LLUVIA_ACUM_7D", "Lluvia acumulada (mm)"),
            use_container_width=True, config=charts.PLOTLY_CONFIG,
        )

        st.markdown("**Variacion del nivel vs Volumen extraido**")
        st.plotly_chart(
            charts.fig_scatter_var(dfv, "EXTRACCION", "Extraccion (lts)"),
            use_container_width=True, config=charts.PLOTLY_CONFIG,
        )

        st.markdown("**Variacion del nivel segun extraccion y lluvia**")
        st.plotly_chart(charts.fig_scatter_2d(dfv), use_container_width=True, config=charts.PLOTLY_CONFIG)

    # --- Tabla completa ---
    with st.expander("Ver tabla de datos"):
        tabla = df[["FECHA", "NIVEL", "LLUVIA", "EXTRACCION"]].sort_values(
            "FECHA", ascending=False
        )
        st.dataframe(
            tabla, hide_index=True, use_container_width=True,
            column_config={"FECHA": st.column_config.DateColumn("FECHA", format="DD/MM/YY")},
        )
