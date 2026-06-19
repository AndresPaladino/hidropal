"""HidroPal - entrypoint Streamlit (mobile-first).

Vista publica: solo-lectura (graficas). Edicion: protegida por PIN recordado
con cookie de ~1 anio. Datos en Supabase. Ver paquete `hidropal/`.
"""
from __future__ import annotations

import os

import streamlit as st

from hidropal import auth, db, styles
from hidropal.config import supabase_enabled
from hidropal.pages_ui import analisis, cargar, registros

st.set_page_config(
    page_title="HidroPal",
    page_icon="logo_pozo.svg",
    initial_sidebar_state="collapsed",
    layout="centered",
)

styles.inject()

_LOGO = os.path.join(os.path.dirname(__file__), "logo_pozo.png")
styles.hero(_LOGO)

# -------------------------
# Pre-requisito: Supabase configurado
# -------------------------
if not supabase_enabled():
    st.error(
        "Falta configurar Supabase. Agrega `[supabase]` en `.streamlit/secrets.toml` "
        "(url y service_key) y crea la tabla con `supabase_schema.sql`."
    )
    st.stop()

auth.init()
editor = auth.is_editor()


# -------------------------
# Acciones del editor (descargar / respaldo / salir)
# -------------------------
def _acciones_editor():
    with st.expander("Acciones y respaldo"):
        st.download_button(
            "Descargar CSV", data=db.export_csv(), file_name="datos_pozo.csv",
            mime="text/csv", width="stretch",
        )
        if st.button("Exportar respaldo a GitHub", width="stretch"):
            ok, msg = db.backup_to_github()
            (st.success if ok else st.error)(msg)
        if st.button("Cerrar sesion", width="stretch"):
            auth.logout()
            st.rerun()


# -------------------------
# Navegacion: control segmentado (estado en session_state -> no se resetea)
# -------------------------
if editor:
    styles.suppress_date_keyboard()
    seccion = st.segmented_control(
        "Navegacion", ["Cargar", "Registros", "Analisis"],
        default="Cargar", label_visibility="collapsed", key="nav",
    )
    if seccion == "Registros":
        registros.render()
    elif seccion == "Analisis":
        analisis.render()
    else:
        cargar.render()
    st.divider()
    _acciones_editor()
else:
    with st.expander("🔒 Ingresar para editar"):
        pin = st.text_input("PIN", type="password", key="login_pin")
        if st.button("Ingresar", type="primary"):
            if auth.try_login(pin):
                st.toast("Sesion iniciada", icon="🔓")
                st.rerun()
            else:
                st.error("PIN incorrecto.")
    analisis.render()
