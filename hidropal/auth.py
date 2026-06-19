"""Autenticacion de edicion: PIN + cookie via streamlit-cookies-controller.

El CookieController debe instanciarse en cada render (init() en app.py) para
que el componente iframe este activo y reporte los valores de las cookies.
En el primer render las cookies no estan disponibles aun (el JS es asincrono);
el componente dispara un rerun automatico cuando las reporta, y en el segundo
render is_editor() ya las lee correctamente.

Las cookies del browser se comparten entre Safari y el modo PWA de iOS para
el mismo dominio, por lo que la sesion persiste al cerrar/reabrir la app.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac

import streamlit as st

from .config import auth_cfg

_COOKIE_NAME = "hidropal_auth"


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
# Init: llamar al inicio de cada render en app.py
# -------------------------
def init():
    """Instancia el CookieController en cada render y lo guarda en session_state.

    Debe llamarse al inicio de app.py en cada render (no dentro de un if/cache)
    para que el componente iframe este presente en el DOM y pueda leer/escribir
    cookies del browser. El componente dispara un rerun automatico la primera
    vez que reporta los valores, por lo que is_editor() tendra las cookies
    disponibles a partir del segundo render.
    """
    from streamlit_cookies_controller import CookieController
    st.session_state["_cc"] = CookieController(key="hidropal_cc")


def _ctrl():
    return st.session_state.get("_cc")


# -------------------------
# API publica
# -------------------------
def is_editor() -> bool:
    """True si la sesion esta autenticada o la cookie contiene un token valido."""
    if st.session_state.get("_is_editor"):
        return True
    cfg = auth_cfg()
    if not cfg:
        return False
    ctrl = _ctrl()
    if ctrl is None:
        return False
    try:
        token = ctrl.get(_COOKIE_NAME) or ""
    except Exception:
        return False
    if token and _token_valid(token, cfg["cookie_secret"]):
        st.session_state["_is_editor"] = True
        return True
    return False


def try_login(pin: str) -> bool:
    """Verifica el PIN; si es correcto escribe la cookie de sesion."""
    cfg = auth_cfg()
    if not cfg:
        return False
    if not hmac.compare_digest(hash_pin(pin.strip()), cfg["pin_hash"]):
        return False
    ctrl = _ctrl()
    if ctrl is None:
        return False
    token = _make_token(cfg["cookie_secret"], cfg["cookie_days"])
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=cfg["cookie_days"])
    try:
        ctrl.set(_COOKIE_NAME, token, expires=expires)
    except Exception:
        pass
    st.session_state["_is_editor"] = True
    return True


def logout() -> None:
    st.session_state["_is_editor"] = False
    ctrl = _ctrl()
    if ctrl:
        try:
            ctrl.remove(_COOKIE_NAME)
        except Exception:
            pass
