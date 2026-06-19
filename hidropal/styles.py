"""CSS mobile-first inyectado. Pensado para un usuario de 65 anios en Safari/iPhone:
targets tactiles grandes, una sola columna, tipografia legible, sin sidebar.
"""
import streamlit as st

_CSS = """
<style>
  :root {
    --hp-blue: #0a3fff;
    --hp-ink: #011887;
    --hp-border: #c9d6ff;
  }

  /* Contenedor mas angosto y con aire en mobile */
  .block-container {
    padding-top: 1rem;
    padding-bottom: 4rem;
    max-width: 680px;
  }

  /* Tipografia mas grande para lectura comoda */
  html, body, [class*="css"] { font-size: 17px; }

  /* Botones grandes y faciles de tocar */
  .stButton > button,
  .stDownloadButton > button,
  .stForm button {
    width: 100%;
    min-height: 3.3rem;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 14px;
  }
  /* Boton primario con un poco de presencia */
  .stButton > button[kind="primary"] {
    box-shadow: 0 6px 16px rgba(10, 63, 255, 0.25);
  }

  /* Inputs con mas contraste (se veian palidos) */
  .stNumberInput input,
  .stDateInput input,
  .stTextInput input {
    min-height: 3.1rem;
    font-size: 1.1rem;
    background: #ffffff !important;
    border: 1.5px solid var(--hp-border) !important;
    border-radius: 12px !important;
    color: var(--hp-ink) !important;
  }
  .stNumberInput label, .stDateInput label, .stTextInput label {
    font-weight: 600; font-size: 1.02rem;
  }
  /* Botones +/- del number_input mas grandes */
  .stNumberInput button { min-height: 3.1rem; min-width: 3rem; }

  /* Tabs como pildoras grandes, scrolleables en horizontal */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    min-height: 3rem;
    padding: 0 16px;
    font-size: 1.02rem;
    font-weight: 600;
    border-radius: 12px;
  }
  .stTabs [aria-selected="true"] {
    background: #e8eeff;
  }

  /* Tarjetas (st.container con border=True) */
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border-color: var(--hp-border) !important;
    box-shadow: 0 8px 24px rgba(1, 24, 135, 0.06);
  }

  /* Tarjetas KPI / resumen (HTML propio) */
  .kpi-row { display: flex; flex-direction: column; gap: 10px; margin: 4px 0 8px; }
  .kpi {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px;
    background: #ffffff;
    border: 1.5px solid var(--hp-border);
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 4px 14px rgba(1, 24, 135, 0.05);
  }
  .kpi-l { display: flex; align-items: center; gap: 11px; }
  .kpi-ico {
    font-size: 1.25rem; line-height: 1;
    width: 40px; height: 40px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    background: #eef3ff;
  }
  .kpi-label { font-weight: 600; font-size: 1rem; color: #44506b; }
  .kpi-r { text-align: right; }
  .kpi-val { font-size: 1.55rem; font-weight: 800; color: var(--hp-ink); line-height: 1.05; }
  .kpi-unit { font-size: 0.95rem; font-weight: 700; color: #8893ab; margin-left: 2px; }
  .kpi-delta { font-size: 0.82rem; color: #64748b; margin-top: 3px; }

  /* Header hero */
  .hidropal-hero {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin: 6px auto 14px;
  }
  .hidropal-logo { width: 56px; height: 56px; }
  .hidropal-logo svg { width: 100%; height: 100%; display: block; }
  .hidropal-title {
    font-size: 2rem; font-weight: 800; margin: 0; line-height: 1.1;
    color: var(--hp-ink);
  }
</style>
"""


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


def metric_cards(items: list[dict]):
    """Renderiza tarjetas KPI/resumen consistentes.

    items: lista de dicts con claves:
      icon (emoji), label (str), value (str), unit (str, opcional),
      delta (str, opcional).
    """
    html = ['<div class="kpi-row">']
    for it in items:
        unit = f'<span class="kpi-unit">{it["unit"]}</span>' if it.get("unit") else ""
        delta = f'<div class="kpi-delta">{it["delta"]}</div>' if it.get("delta") else ""
        html.append(
            '<div class="kpi">'
            f'<div class="kpi-l"><span class="kpi-ico">{it.get("icon", "")}</span>'
            f'<span class="kpi-label">{it["label"]}</span></div>'
            f'<div class="kpi-r"><div class="kpi-val">{it["value"]}{unit}</div>{delta}</div>'
            "</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def hero(svg_icon: str):
    st.markdown(
        f"""
        <div class="hidropal-hero">
          <div class="hidropal-logo">{svg_icon}</div>
          <h1 class="hidropal-title">HidroPal</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
