"""Lenguaje visual Apple-refined (iOS) para HidroPal.

Paleta neutral: fondo gris iOS, tarjetas blancas con esquinas grandes y sombra
sutil, texto casi negro, secundario gris, y el azul de marca solo como acento.
Pensado mobile-first para Safari/iPhone (usuario de 65).
"""
import streamlit as st
import streamlit.components.v1 as components

_CSS = """
<style>
  :root {
    --hp-bg: #F2F2F7;          /* gris iOS de fondo */
    --hp-card: #FFFFFF;
    --hp-ink: #1C1C1E;         /* texto casi negro */
    --hp-sub: #8E8E93;         /* gris secundario */
    --hp-line: #E5E5EA;        /* hairline */
    --hp-accent: #0A3FFF;      /* azul de marca (acento) */
    --hp-accent-soft: #EAEFFF;
    --hp-red: #FF3B30;         /* rojo sistema */
    --hp-radius: 18px;
  }

  /* Tipografia del sistema (SF en iOS) */
  html, body, [class*="css"], .stMarkdown, button, input {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
      Roboto, Helvetica, Arial, sans-serif;
  }
  html, body { font-size: 17px; }

  /* Ocultar barra de Streamlit */
  header[data-testid="stHeader"], #stDecoration { display: none !important; }

  /* Contenedor angosto */
  .block-container {
    padding-top: 1.2rem;
    padding-bottom: 4rem;
    max-width: 640px;
  }

  /* Titulos mas suaves */
  h1, h2, h3 { color: var(--hp-ink); letter-spacing: -0.02em; }
  .stMarkdown h2, .stMarkdown h3 { font-weight: 700; }

  /* ---- Header slim ---- */
  .hidropal-hero {
    display: flex; align-items: center; justify-content: center;
    gap: 9px; margin: 2px auto 10px;
  }
  .hidropal-logo { width: 26px; height: 26px; line-height: 0; }
  .hidropal-logo svg { width: 100%; height: 100%; display: block; }
  .hidropal-title {
    font-size: 1.25rem; font-weight: 700; margin: 0; color: var(--hp-ink);
    letter-spacing: -0.02em;
  }

  /* ---- Tarjetas (st.container border=True) estilo iOS grouped ---- */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--hp-card);
    border: 1px solid var(--hp-line) !important;
    border-radius: var(--hp-radius) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    padding: 4px 4px;
  }

  /* ---- Inputs ----
     El borde vive en el wrapper de BaseWeb (no en el <input> interno) para
     evitar el borde doble: el wrapper ya trae su propio borde y el input otro. */
  .stNumberInput [data-baseweb="input"],
  .stNumberInput [data-baseweb="base-input"],
  .stDateInput [data-baseweb="input"],
  .stTextInput [data-baseweb="input"] {
    background: #fff !important;
    border: 1px solid var(--hp-line) !important;
    border-radius: 12px !important;
    overflow: hidden;
    transition: border-color 150ms ease-out, box-shadow 150ms ease-out;
  }
  .stNumberInput input, .stDateInput input, .stTextInput input {
    min-height: 3rem;
    font-size: 1.08rem;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--hp-ink) !important;
  }
  /* Anillo de foco una sola vez, en el wrapper */
  .stNumberInput [data-baseweb="input"]:focus-within,
  .stDateInput [data-baseweb="input"]:focus-within,
  .stTextInput [data-baseweb="input"]:focus-within {
    border-color: var(--hp-accent) !important;
    box-shadow: 0 0 0 3px rgba(10,63,255,0.15) !important;
  }
  .stNumberInput label, .stDateInput label, .stTextInput label,
  .stRadio label, .stPills label {
    font-weight: 600; font-size: 0.98rem; color: var(--hp-ink);
  }
  .stNumberInput button { min-height: 3rem; min-width: 2.8rem; }

  /* ---- Botones ---- */
  .stButton > button, .stDownloadButton > button, .stForm button {
    width: 100%;
    min-height: 3.2rem;
    font-size: 1.08rem;
    font-weight: 600;
    border-radius: 14px;
    border: 1px solid var(--hp-line);
    background: #fff;
    color: var(--hp-accent);
    transition: transform 120ms cubic-bezier(0.23, 1, 0.32, 1);
  }
  .stButton > button:active { transform: scale(0.98); }
  /* Primario: azul lleno */
  .stButton > button[kind="primary"], .stForm button[kind="primaryFormSubmit"] {
    background: var(--hp-accent);
    border-color: var(--hp-accent);
    color: #fff;
    box-shadow: 0 6px 16px rgba(10,63,255,0.22);
  }
  /* Destructivo (botones con key=btn_eliminar / btn_purge) */
  .st-key-btn_eliminar button, .st-key-btn_purge button {
    background: #fff; color: var(--hp-red); border-color: rgba(255,59,48,0.35);
  }
  .st-key-btn_eliminar button:active, .st-key-btn_purge button:active {
    background: rgba(255,59,48,0.06);
  }

  /* ---- Segmented control (navegacion) estilo iOS ---- */
  [data-testid="stSegmentedControl"] [role="group"] {
    background: #E3E3E8;
    border-radius: 12px;
    padding: 3px;
    gap: 2px;
    width: 100%;
  }
  [data-testid="stSegmentedControl"] button {
    border: none !important;
    background: transparent !important;
    border-radius: 9px !important;
    min-height: 2.6rem;
    color: var(--hp-ink) !important;
    font-weight: 600 !important;
    flex: 1;
    transition: background 150ms ease-out, box-shadow 150ms ease-out;
  }
  [data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: #fff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  }

  /* ---- Pills (comparacion) ---- */
  [data-testid="stPills"] button[aria-checked="true"] {
    background: var(--hp-accent-soft) !important;
    border-color: var(--hp-accent) !important;
    color: var(--hp-accent) !important;
  }

  /* ---- Tarjetas KPI / resumen (HTML propio) ---- */
  .kpi-row { display: flex; flex-direction: column; gap: 9px; margin: 4px 0 8px; }
  .kpi {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    background: var(--hp-card);
    border: 1px solid var(--hp-line);
    border-radius: var(--hp-radius);
    padding: 13px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .kpi-l { display: flex; align-items: center; gap: 11px; }
  .kpi-ico {
    font-size: 1.15rem; line-height: 1;
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    background: var(--hp-accent-soft);
  }
  .kpi-label { font-weight: 500; font-size: 0.98rem; color: var(--hp-sub); }
  .kpi-r { text-align: right; }
  .kpi-val { font-size: 1.5rem; font-weight: 700; color: var(--hp-ink); line-height: 1.05; }
  .kpi-unit { font-size: 0.9rem; font-weight: 600; color: var(--hp-sub); margin-left: 3px; }
  .kpi-delta { font-size: 0.8rem; color: var(--hp-sub); margin-top: 3px; }

  /* Dataframe: bordes suaves */
  [data-testid="stDataFrame"] { border-radius: 14px; }

  hr { border-color: var(--hp-line); }

  @media (prefers-reduced-motion: reduce) {
    .stButton > button, .stDownloadButton > button, .stForm button {
      transition: none;
    }
    [data-testid="stSegmentedControl"] button {
      transition: none;
    }
    .stNumberInput [data-baseweb="input"],
    .stDateInput [data-baseweb="input"],
    .stTextInput [data-baseweb="input"] {
      transition: none;
    }
  }
</style>
"""


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)


def suppress_date_keyboard():
    """Evita el teclado virtual en los date_input (iOS): setea inputmode=none
    en los inputs de fecha del documento padre (same-origin) y reaplica tras
    reruns con un MutationObserver. El calendario sigue funcionando.
    """
    components.html(
        """
        <script>
        try {
          const doc = window.parent.document;
          function fix() {
            doc.querySelectorAll('[data-testid="stDateInput"] input').forEach(el => {
              el.setAttribute('inputmode', 'none');
              el.setAttribute('autocomplete', 'off');
            });
          }
          fix();
          new MutationObserver(fix).observe(doc.body, {childList: true, subtree: true});
        } catch(e) {}
        </script>
        """,
        height=0,
        scrolling=False,
    )


def hero(logo_path: str | None = None):
    import base64
    logo_html = ""
    if logo_path:
        try:
            with open(logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            logo_html = (
                f'<img src="data:image/png;base64,{b64}" '
                'style="width:28px;height:28px;object-fit:contain">'
            )
        except OSError:
            pass
    st.markdown(
        f"""
        <div class="hidropal-hero">
          {logo_html}
          <span class="hidropal-title">HidroPal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(items: list[dict]):
    """Renderiza tarjetas KPI/resumen consistentes.

    items: lista de dicts con claves icon, label, value, unit (opc), delta (opc).
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
