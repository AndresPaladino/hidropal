"""CSS mobile-first inyectado. Pensado para un usuario de 65 anios en Safari/iPhone:
targets tactiles grandes, una sola columna, tipografia legible, sin sidebar.
"""
import streamlit as st

_CSS = """
<style>
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
  .stDownloadButton > button {
    width: 100%;
    min-height: 3.2rem;
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 14px;
  }

  /* Inputs altos */
  .stNumberInput input,
  .stDateInput input,
  .stTextInput input {
    min-height: 3rem;
    font-size: 1.05rem;
  }

  /* Tabs como pildoras grandes, scrolleables en horizontal */
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    min-height: 3rem;
    padding: 0 14px;
    font-size: 1rem;
  }

  /* Metricas centradas */
  [data-testid="stMetric"] { text-align: center; }

  /* Header hero */
  .hidropal-hero {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin: 6px auto 14px;
  }
  .hidropal-logo { width: 56px; height: 56px; }
  .hidropal-logo svg { width: 100%; height: 100%; display: block; }
  .hidropal-title {
    font-size: 2rem; font-weight: 800; margin: 0; line-height: 1.1;
    color: #011887;
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
