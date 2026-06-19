"""Autenticacion de edicion: PIN + cookie via streamlit-cookies-controller.

Modelo: la vista publica es de solo-lectura. Para editar hay que ingresar un
PIN una sola vez. El token HMAC firmado se guarda en una cookie del browser
via streamlit-cookies-controller (usa document.cookie desde un iframe con
allow-same-origin). Las cookies del browser se comparten entre Safari y el
modo PWA (iPhone home screen), por lo que la sesion persiste al cerrar y
reabrir la app desde la pantalla de inicio.
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
# Cookie manager (singleton por sesion)
# -------------------------
def _get_controller():
    if "_cookie_ctrl" not in st.session_state:
        from streamlit_cookies_controller import CookieController
        st.session_state["_cookie_ctrl"] = CookieController(key="hidropal_cc")
    return st.session_state["_cookie_ctrl"]


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
    try:
        token = _get_controller().get(_COOKIE_NAME) or ""
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
    token = _make_token(cfg["cookie_secret"], cfg["cookie_days"])
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=cfg["cookie_days"])
    try:
        _get_controller().set(_COOKIE_NAME, token, expires=expires)
    except Exception:
        pass
    st.session_state["_is_editor"] = True
    return True


def logout() -> None:
    st.session_state["_is_editor"] = False
    try:
        _get_controller().remove(_COOKIE_NAME)
    except Exception:
        pass
