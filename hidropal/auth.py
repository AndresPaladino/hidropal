"""Autenticacion de edicion: PIN + token firmado en query params.

Modelo: la vista publica es de solo-lectura. Para editar hay que ingresar un
PIN una sola vez; se guarda un token HMAC en st.query_params (persiste en la
URL del browser entre recargas sin depender de cookies ni iframes JS).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac

import streamlit as st

from .config import auth_cfg

_PARAM = "t"


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
    """Verifica el PIN; si es correcto escribe el token en la URL."""
    cfg = auth_cfg()
    if not cfg:
        return False
    if not hmac.compare_digest(hash_pin(pin.strip()), cfg["pin_hash"]):
        return False
    token = _make_token(cfg["cookie_secret"], cfg["cookie_days"])
    st.query_params[_PARAM] = token
    st.session_state["_is_editor"] = True
    return True


def logout() -> None:
    st.session_state["_is_editor"] = False
    st.query_params.pop(_PARAM, None)
