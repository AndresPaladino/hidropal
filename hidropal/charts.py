"""Constructores de graficas en Plotly (responsive / mobile-first).

Reconstruye las mismas graficas del app original (mismos datos, colores y el
eje de nivel invertido), pero interactivas y a ancho completo en el celular.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLOR_PALETTE as C

_FONT = dict(
    family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", '
    "Roboto, Helvetica, Arial, sans-serif",
    color="#1C1C1E",
    size=13,
)
_GRID = "#E5E5EA"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(orientation="h", y=-0.2),
    font=_FONT,
    colorway=["#0A3FFF"],
    dragmode=False,  # mobile: que el dedo no haga pan/zoom dentro del grafico
)

# Config para st.plotly_chart: sin barra de herramientas ni zoom por scroll.
# Se mantiene el tap (tooltip) pero el grafico no pelea con el scroll de la pagina.
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "displaylogo": False,
}


def filtrar_rango(df: pd.DataFrame, dias: int | None) -> pd.DataFrame:
    """Filtra a los ultimos `dias` desde el ultimo registro. None = todo."""
    if dias is None or df.empty:
        return df
    corte = df["FECHA"].max() - pd.Timedelta(days=dias)
    return df[df["FECHA"] >= corte]


def _clean(fig):
    """Grilla tenue y ejes sin linea cero, en todos los subplots."""
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


def _line(df, col, color, name, mode="lines+markers"):
    return go.Scatter(
        x=df["FECHA"], y=df[col], mode=mode, name=name,
        line=dict(color=color), marker=dict(color=color),
    )


def _bar(df, col, color, name):
    return go.Bar(x=df["FECHA"], y=df[col], name=name, marker_color=color)


# =========================================================================
# Graficas principales (curadas, mobile-first)
# =========================================================================

# A) Nivel del pozo: linea, eje invertido (mas arriba = mas agua, pedido del usuario)
def fig_nivel(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(_line(df, "NIVEL", C["NIVEL"], "Nivel"))
    fig.update_yaxes(autorange="reversed", title_text="Nivel (m)")
    fig.update_layout(height=340, showlegend=False, **_LAYOUT)
    return _clean(fig)


# B) Nivel + Lluvia en un solo grafico con DOBLE eje Y (escalas reales, no
#    normalizadas): responde "¿la lluvia recupera el pozo?" sin engañar.
def fig_nivel_lluvia(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(_bar(df, "LLUVIA", C["LLUVIA"], "Lluvia (mm)"), secondary_y=True)
    fig.add_trace(
        _line(df, "NIVEL", C["NIVEL"], "Nivel (m)"), secondary_y=False
    )
    # Nivel invertido a la izquierda; lluvia normal a la derecha, sin grilla.
    fig.update_yaxes(autorange="reversed", title_text="Nivel (m)", secondary_y=False)
    fig.update_yaxes(title_text="Lluvia (mm)", secondary_y=True, showgrid=False)
    fig.update_layout(height=380, **_LAYOUT)
    return _clean(fig)


# C) Extraccion en el tiempo: barras simples.
def fig_extraccion(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(_bar(df, "EXTRACCION", C["EXTRACCION"], "Extraccion"))
    fig.update_layout(height=300, showlegend=False, yaxis_title="Litros", **_LAYOUT)
    return _clean(fig)


# =========================================================================
# Graficas avanzadas (plegadas en "Mas analisis")
# =========================================================================

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
    return _clean(fig)


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
    return _clean(fig)


# 3) Comparacion de tendencias ADAPTATIVA segun cuantas variables se eligen:
#    1 -> escala real | 2 -> doble eje (escalas reales) | 3+ -> normalizado.
#    'invert' solo aplica a Nivel (mas arriba = mas agua, pedido del usuario).
_COMPARABLES = {
    "Nivel": dict(col="NIVEL", color=C["NIVEL"], unit="m", kind="line", invert=True),
    "Lluvia": dict(col="LLUVIA", color=C["LLUVIA"], unit="mm", kind="bar", invert=False),
    "Extraccion": dict(col="EXTRACCION", color=C["EXTRACCION"], unit="lts", kind="bar", invert=False),
    "Variacion de nivel": dict(col="VARIACION_NIVEL", color=C["VARIACION_NIVEL"], unit="m", kind="line", invert=False),
    "Lluvia acum. 7d": dict(col="LLUVIA_ACUM_7D", color=C["LLUVIA_ACUM_7D"], unit="mm", kind="line", invert=False),
}


def opciones_comparar() -> list[str]:
    """Nombres disponibles para el selector del comparador."""
    return list(_COMPARABLES)


def _comp_trace(df, name, y=None, force_line=False, **extra):
    """Traza (linea o barra) con tooltip que SIEMPRE muestra el valor real,
    aunque `y` venga normalizada (customdata = serie real)."""
    m = _COMPARABLES[name]
    real = df[m["col"]]
    y = real if y is None else y
    ht = f"%{{x|%d/%m/%Y}}<br>{name}: %{{customdata:.2f}} {m['unit']}<extra></extra>"
    common = dict(x=df["FECHA"], name=name, customdata=real, hovertemplate=ht)
    if m["kind"] == "bar" and not force_line:
        return go.Bar(y=y, marker_color=m["color"], **common, **extra)
    return go.Scatter(
        y=y, mode="lines+markers", line=dict(color=m["color"]),
        marker=dict(color=m["color"]), **common, **extra,
    )


def _yaxis_kwargs(name, secondary=False):
    m = _COMPARABLES[name]
    kw = {"title_text": f"{name} ({m['unit']})"}
    if secondary:
        kw["secondary_y"] = True
        kw["showgrid"] = False
    if m["invert"]:
        kw["autorange"] = "reversed"
    return kw


def fig_comparacion(df: pd.DataFrame, seleccion: list[str]) -> go.Figure:
    sel = [s for s in seleccion if s in _COMPARABLES]
    if not sel:
        return go.Figure()

    # 1 variable: escala real, una sola.
    if len(sel) == 1:
        fig = go.Figure(_comp_trace(df, sel[0]))
        fig.update_yaxes(**_yaxis_kwargs(sel[0]))
        fig.update_layout(height=360, showlegend=False, **_LAYOUT)
        return _clean(fig)

    # 2 variables: doble eje, magnitudes reales (honesto).
    if len(sel) == 2:
        left, right = sel
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(_comp_trace(df, right), secondary_y=True)
        fig.add_trace(_comp_trace(df, left), secondary_y=False)
        _clean(fig)
        fig.update_yaxes(**_yaxis_kwargs(left))
        fig.update_yaxes(**_yaxis_kwargs(right, secondary=True))
        fig.update_layout(height=380, **_LAYOUT)
        return fig

    # 3+ variables: overlay normalizado 0-100% (compara la FORMA, no la
    # magnitud); el tooltip sigue mostrando el valor real al tocar.
    fig = go.Figure()
    for name in sel:
        m = _COMPARABLES[name]
        serie = -df[m["col"]] if m["invert"] else df[m["col"]]
        rng = serie.max() - serie.min()
        norm = (serie - serie.min()) / rng * 100 if rng else serie * 0
        fig.add_trace(_comp_trace(df, name, y=norm, force_line=True))
    fig.update_yaxes(title_text="Relativo (0-100%)")
    fig.update_layout(height=420, **_LAYOUT)
    return _clean(fig)


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
    return _clean(fig)


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
    return _clean(fig)
