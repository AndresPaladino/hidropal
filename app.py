"""HidroPal - entrypoint Streamlit (mobile-first).

Vista publica: solo-lectura (graficas). Edicion: protegida por PIN recordado
con cookie de ~1 anio. Datos en Supabase. Ver paquete `hidropal/`.
"""
from __future__ import annotations

import os

import streamlit as st

from hidropal import auth, db, styles
from hidropal.config import supabase_enabled
from hidropal.pages_ui import analisis, cargar, eliminar, modificar, restaurar

st.set_page_config(
    page_title="HidroPal",
    page_icon="logo_pozo.svg",
    initial_sidebar_state="collapsed",
    layout="centered",
)

styles.inject()

_LOGO = os.path.join(os.path.dirname(__file__), "logo_grande.svg")
try:
    with open(_LOGO, "r") as f:
        styles.hero(f.read())
except OSError:
    st.title("HidroPal")

# -------------------------
# Pre-requisito: Supabase configurado
# -------------------------
if not supabase_enabled():
    st.error(
        "Falta configurar Supabase. Agrega `[supabase]` en `.streamlit/secrets.toml` "
        "(url y service_key) y crea la tabla con `supabase_schema.sql`."
    )
    st.stop()

editor = auth.is_editor()


# -------------------------
# Acciones del editor (descargar / respaldo / salir)
# -------------------------
def _acciones_editor():
    with st.expander("Acciones y respaldo"):
        st.download_button(
            "Descargar CSV", data=db.export_csv(), file_name="datos_pozo.csv",
            mime="text/csv", use_container_width=True,
        )
        if st.button("Exportar respaldo a GitHub", use_container_width=True):
            ok, msg = db.backup_to_github()
            (st.success if ok else st.error)(msg)
        if st.button("Cerrar sesion", use_container_width=True):
            auth.logout()
            st.rerun()


# -------------------------
# Navegacion
# -------------------------
if editor:
    tab_analisis, tab_datos = st.tabs(["Analisis", "Datos"])
    with tab_analisis:
        analisis.render()
    with tab_datos:
        sub = st.tabs(["Cargar", "Modificar", "Eliminar", "Papelera"])
        with sub[0]:
            cargar.render()
        with sub[1]:
            modificar.render()
        with sub[2]:
            eliminar.render()
        with sub[3]:
            restaurar.render()
        st.divider()
        _acciones_editor()
else:
    analisis.render()
    st.divider()
    with st.expander("Ingresar para editar"):
        pin = st.text_input("PIN", type="password", key="login_pin")
        if st.button("Ingresar", type="primary"):
            if auth.try_login(pin):
                st.toast("Sesion iniciada", icon="🔓")
                st.rerun()
            else:
                st.error("PIN incorrecto.")
