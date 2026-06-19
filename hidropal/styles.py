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

  /* Metricas: numeros grandes, centradas, como tarjeta */
  [data-testid="stMetric"] {
    text-align: center;
    background: #ffffff;
    border: 1.5px solid var(--hp-border);
    border-radius: 16px;
    padding: 12px 6px;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.5rem; font-weight: 800; color: var(--hp-ink);
  }
  [data-testid="stMetricLabel"] { justify-content: center; }

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
