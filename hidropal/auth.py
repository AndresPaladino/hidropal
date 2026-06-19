"""Autenticacion de edicion: PIN + token firmado en query params + localStorage.

Modelo: la vista publica es de solo-lectura. Para editar hay que ingresar un
PIN una sola vez. El token HMAC se guarda en:
  - st.query_params: persiste en la URL entre recargas en el browser normal.
  - localStorage: persiste entre sesiones en modo PWA (iPhone home screen),
    donde la app siempre abre la URL original sin query params.

inject_session_restore() debe llamarse al inicio de cada render para sincronizar
ambos stores y redirigir si el token esta en localStorage pero no en la URL.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac

import streamlit as st
import streamlit.components.v1 as components

from .config import auth_cfg

_PARAM = "t"
_LS_KEY = "hidropal_auth"


# -------------------------
# Hash / firma
# -------------------------
def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def _sign(value: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _make_token(secret: str, days: int) -> str:
    exp = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)
    exp_iso = exp.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{exp_iso}.{_sign(exp_iso, secret)}"


def _token_valid(token: str, secret: str) -> bool:
    try:
        exp_iso, sig = token.rsplit(".", 1)
    except ValueError:
        return False
    if not hmac.compare_digest(sig, _sign(exp_iso, secret)):
        return False
    try:
        exp = dt.datetime.strptime(exp_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=dt.timezone.utc
        )
    except ValueError:
        return False
    return exp > dt.datetime.now(dt.timezone.utc)


# -------------------------
# Sincronizacion localStorage <-> URL (para modo PWA)
# -------------------------
def inject_session_restore():
    """Sincroniza el token entre la URL y localStorage en cada render.

    - URL tiene token → lo copia a localStorage (backup para modo PWA).
    - URL sin token pero localStorage si → redirige con el token en la URL.

    En modo PWA (iPhone home screen) la app siempre abre la URL guardada al
    momento de "Agregar al inicio", sin query params. Este script recupera la
    sesion desde localStorage y la inyecta en la URL para que is_editor() la
    encuentre en el siguiente render.
    """
    components.html(
        f"""
        <script>
        (function() {{
          try {{
            var url = new URL(window.parent.location.href);
            var urlToken = url.searchParams.get('{_PARAM}');
            var lsToken = localStorage.getItem('{_LS_KEY}');
            if (urlToken) {{
              localStorage.setItem('{_LS_KEY}', urlToken);
            }} else if (lsToken) {{
              url.searchParams.set('{_PARAM}', lsToken);
              window.parent.location.replace(url.toString());
            }}
          }} catch(e) {{}}
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )


# -------------------------
# API publica
# -------------------------
def is_editor() -> bool:
    """True si la sesion esta autenticada o el query param contiene un token valido."""
    if st.session_state.get("_is_editor"):
        return True
    cfg = auth_cfg()
    if not cfg:
        return False
    token = st.query_params.get(_PARAM, "")
    if token and _token_valid(token, cfg["cookie_secret"]):
        st.session_state["_is_editor"] = True
        return True
    return False


def try_login(pin: str) -> bool:
    """Verifica el PIN; si es correcto escribe el token en la URL y localStorage."""
    cfg = auth_cfg()
    if not cfg:
        return False
    if not hmac.compare_digest(hash_pin(pin.strip()), cfg["pin_hash"]):
        return False
    token = _make_token(cfg["cookie_secret"], cfg["cookie_days"])
    st.query_params[_PARAM] = token
    # localStorage se sincroniza via inject_session_restore() en el proximo render
    st.session_state["_is_editor"] = True
    return True


def logout() -> None:
    st.session_state["_is_editor"] = False
    st.query_params.pop(_PARAM, None)
    components.html(
        f"<script>try{{localStorage.removeItem('{_LS_KEY}')}}catch(e){{}}</script>",
        height=0,
        scrolling=False,
    )
