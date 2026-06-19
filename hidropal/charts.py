"""Graficas con matplotlib (igual a la version original)."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .config import COLOR_PALETTE as C

_DATE_FMT = mdates.DateFormatter("%d/%m/%Y")


def filtrar_rango(df: pd.DataFrame, dias: int | None) -> pd.DataFrame:
    if dias is None or df.empty:
        return df
    corte = df["FECHA"].max() - pd.Timedelta(days=dias)
    return df[df["FECHA"] >= corte]


def _fmt_dates(axes):
    for ax in axes:
        ax.xaxis.set_major_formatter(_DATE_FMT)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")


def fig_serie_temporal(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_alpha(0.0)

    axes[0].plot(df["FECHA"], df["NIVEL"], marker="o", color=C["NIVEL"])
    axes[0].invert_yaxis()
    axes[0].set_title("Nivel de Agua")
    axes[0].set_ylabel("Nivel (m)")
    axes[0].grid(True)
    axes[0].patch.set_alpha(0.0)

    axes[1].plot(df["FECHA"], df["LLUVIA"], marker="o", color=C["LLUVIA"])
    axes[1].set_title("Lluvia Caída")
    axes[1].set_ylabel("mm")
    axes[1].grid(True)
    axes[1].patch.set_alpha(0.0)

    axes[2].bar(df["FECHA"], df["EXTRACCION"], color=C["EXTRACCION"])
    axes[2].set_title("Volumen Extraído")
    axes[2].set_ylabel("Litros")
    axes[2].set_xlabel("Fecha")
    axes[2].grid(True)
    axes[2].patch.set_alpha(0.0)

    _fmt_dates(axes)
    return fig


def fig_dashboard(df: pd.DataFrame):
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_alpha(0.0)

    axes[0].plot(df["FECHA"], df["NIVEL"], marker="o", color=C["NIVEL"])
    axes[0].invert_yaxis()
    axes[0].set_title("Nivel de Agua")
    axes[0].grid(True)
    axes[0].patch.set_alpha(0.0)

    axes[1].plot(df["FECHA"], df["LLUVIA"], marker="o", color=C["LLUVIA"])
    axes[1].set_title("Lluvia Caída")
    axes[1].set_ylabel("mm")
    axes[1].grid(True)
    axes[1].patch.set_alpha(0.0)

    axes[2].bar(df["FECHA"], df["VARIACION_NIVEL"], color=C["VARIACION_NIVEL"])
    axes[2].axhline(0, color="k", linestyle="--")
    axes[2].set_title("Variación del Nivel")
    axes[2].grid(True)
    axes[2].patch.set_alpha(0.0)

    axes[3].plot(df["FECHA"], df["LLUVIA_ACUM_7D"], marker="o", color=C["LLUVIA_ACUM_7D"])
    axes[3].set_title("Lluvia acumulada 7 días")
    axes[3].grid(True)
    axes[3].patch.set_alpha(0.0)

    axes[4].bar(df["FECHA"], df["EXTRACCION"], color=C["EXTRACCION"])
    axes[4].set_title("Volumen extraído")
    axes[4].set_xlabel("Fecha")
    axes[4].grid(True)
    axes[4].patch.set_alpha(0.0)

    _fmt_dates(axes)
    return fig


def opciones_comparar() -> list[str]:
    return ["Nivel", "Lluvia", "Extracción", "Variación de nivel", "Lluvia Acumulada (7 dias)"]


def fig_comparacion(df: pd.DataFrame, seleccion: list[str]):
    variables = {
        "Nivel": -df["NIVEL"],
        "Lluvia": df["LLUVIA"],
        "Extracción": df["EXTRACCION"],
        "Variación de nivel": df["VARIACION_NIVEL"],
        "Lluvia Acumulada (7 dias)": df["LLUVIA_ACUM_7D"],
    }
    var_colors = {
        "Nivel": C["NIVEL"],
        "Lluvia": C["LLUVIA"],
        "Extracción": C["EXTRACCION"],
        "Variación de nivel": C["VARIACION_NIVEL"],
        "Lluvia Acumulada (7 dias)": C["LLUVIA_ACUM_7D"],
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for var in seleccion:
        if var not in variables:
            continue
        serie = variables[var]
        rng = serie.max() - serie.min()
        serie_norm = (serie - serie.min()) / rng if rng else serie * 0
        ax.plot(df["FECHA"], serie_norm, label=var, marker="o", color=var_colors[var])

    _fmt_dates([ax])
    ax.legend()
    return fig


def fig_scatter_var_lluvia(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.scatter(df["LLUVIA_ACUM_7D"], df["VARIACION_NIVEL"], color=C["VARIACION_NIVEL"], alpha=0.6)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Lluvia acumulada (mm)")
    ax.set_ylabel("ΔNivel (m)")
    ax.grid(True)
    return fig


def fig_scatter_var_extraccion(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.scatter(df["EXTRACCION"], df["VARIACION_NIVEL"], color=C["VARIACION_NIVEL"], alpha=0.6)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Extracción (lts)")
    ax.set_ylabel("ΔNivel (m)")
    ax.grid(True)
    return fig


def fig_scatter_2d(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    scatter = ax.scatter(
        df["EXTRACCION"], df["LLUVIA_ACUM_7D"],
        c=df["VARIACION_NIVEL"], cmap="inferno",
    )
    plt.colorbar(scatter, label="ΔNivel (m)", ax=ax)
    ax.set_xlabel("Extracción (lts)")
    ax.set_ylabel("Lluvia acumulada (mm)")
    ax.grid(True)
    return fig
