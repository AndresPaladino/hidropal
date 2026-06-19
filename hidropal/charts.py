"""Constructores de graficas en Plotly (responsive / mobile-first).

Reconstruye las mismas graficas del app original (mismos datos, colores y el
eje de nivel invertido), pero interactivas y a ancho completo en el celular.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLOR_PALETTE as C

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", y=-0.2),
)


def _line(df, col, color, name, mode="lines+markers"):
    return go.Scatter(
        x=df["FECHA"], y=df[col], mode=mode, name=name,
        line=dict(color=color), marker=dict(color=color),
    )


def _bar(df, col, color, name):
    return go.Bar(x=df["FECHA"], y=df[col], name=name, marker_color=color)


# 1) Serie temporal: nivel (eje invertido), lluvia, extraccion
def fig_serie_temporal(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Nivel de agua (m)", "Lluvia caida (mm)", "Volumen extraido (lts)"),
    )
    fig.add_trace(_line(df, "NIVEL", C["NIVEL"], "Nivel"), row=1, col=1)
    fig.add_trace(_line(df, "LLUVIA", C["LLUVIA"], "Lluvia"), row=2, col=1)
    fig.add_trace(_bar(df, "EXTRACCION", C["EXTRACCION"], "Extraccion"), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)  # nivel invertido
    fig.update_layout(height=620, showlegend=False, **_LAYOUT)
    return fig


# 2) Dashboard de 5 paneles
def fig_dashboard(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            "Nivel de agua (m)", "Lluvia caida (mm)", "Variacion del nivel",
            "Lluvia acumulada 7 dias (mm)", "Volumen extraido (lts)",
        ),
    )
    fig.add_trace(_line(df, "NIVEL", C["NIVEL"], "Nivel"), row=1, col=1)
    fig.add_trace(_line(df, "LLUVIA", C["LLUVIA"], "Lluvia"), row=2, col=1)
    fig.add_trace(_bar(df, "VARIACION_NIVEL", C["VARIACION_NIVEL"], "Variacion"), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)
    fig.add_trace(_line(df, "LLUVIA_ACUM_7D", C["LLUVIA_ACUM_7D"], "Lluvia 7d"), row=4, col=1)
    fig.add_trace(_bar(df, "EXTRACCION", C["EXTRACCION"], "Extraccion"), row=5, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)  # nivel invertido
    fig.update_layout(height=900, showlegend=False, **_LAYOUT)
    return fig


# 3) Comparacion de tendencias (multiselect, normalizado min-max)
def variables_para_comparar(df: pd.DataFrame) -> dict:
    """Mapa nombre -> serie. 'Nivel' usa -NIVEL (igual que el original)."""
    return {
        "Nivel": -df["NIVEL"],
        "Lluvia": df["LLUVIA"],
        "Extraccion": df["EXTRACCION"],
        "Variacion de nivel": df["VARIACION_NIVEL"],
        "Lluvia acumulada (7 dias)": df["LLUVIA_ACUM_7D"],
    }


_VAR_COLOR = {
    "Nivel": C["NIVEL"],
    "Lluvia": C["LLUVIA"],
    "Extraccion": C["EXTRACCION"],
    "Variacion de nivel": C["VARIACION_NIVEL"],
    "Lluvia acumulada (7 dias)": C["LLUVIA_ACUM_7D"],
}


def fig_comparacion(df: pd.DataFrame, seleccion: list[str]) -> go.Figure:
    variables = variables_para_comparar(df)
    fig = go.Figure()
    for var in seleccion:
        serie = variables[var]
        rng = serie.max() - serie.min()
        norm = (serie - serie.min()) / rng if rng else serie * 0
        fig.add_trace(go.Scatter(
            x=df["FECHA"], y=norm, mode="lines+markers", name=var,
            line=dict(color=_VAR_COLOR[var]), marker=dict(color=_VAR_COLOR[var]),
        ))
    fig.update_layout(height=420, **_LAYOUT)
    return fig


# 4-5) Scatters de variacion vs (lluvia7d / extraccion)
def fig_scatter_var(df: pd.DataFrame, x_col: str, x_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df["VARIACION_NIVEL"], mode="markers",
        marker=dict(color=C["VARIACION_NIVEL"], opacity=0.7, size=9),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        height=420, xaxis_title=x_label, yaxis_title="dNivel (m)", **_LAYOUT
    )
    return fig


# 6) Scatter 2D extraccion x lluvia7d, color por variacion (colorbar)
def fig_scatter_2d(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["EXTRACCION"], y=df["LLUVIA_ACUM_7D"], mode="markers",
        marker=dict(
            color=df["VARIACION_NIVEL"], colorscale="Inferno", size=11,
            colorbar=dict(title="dNivel (m)"), showscale=True,
        ),
    ))
    fig.update_layout(
        height=460, xaxis_title="Extraccion (lts)",
        yaxis_title="Lluvia acumulada (mm)", **_LAYOUT,
    )
    return fig
