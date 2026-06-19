"""Autenticacion de edicion: PIN + cookie firmada de larga duracion.

Modelo: la vista publica es de solo-lectura. Para editar hay que ingresar un
PIN una sola vez; se guarda una cookie firmada (HMAC) que dura ~1 anio, asi el
usuario no vuelve a loguearse en cada visita.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac

import streamlit as st

from .config import auth_cfg


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
# Cookie manager
# -------------------------
def init_cookies():
    """Construye el CookieManager una vez por sesion y fuerza un rerun inicial.

    El CookieManager lee las cookies del browser via JS de forma asincrona.
    En el primer render no tiene los valores aun; el rerun garantiza que el
    segundo render ya los tenga disponibles.
    """
    import extra_streamlit_components as stx

    first_run = "_cookie_mgr" not in st.session_state
    st.session_state["_cookie_mgr"] = stx.CookieManager(key="hidropal_cookies")
    if first_run:
        st.rerun()


def _cookies():
    cm = st.session_state.get("_cookie_mgr")
    if cm is None:
        init_cookies()
        cm = st.session_state["_cookie_mgr"]
    return cm


# -------------------------
# API publica
# -------------------------
def is_editor() -> bool:
    """True si la sesion tiene una cookie valida o se autentico en esta sesion."""
    if st.session_state.get("_is_editor"):
        return True
    cfg = auth_cfg()
    if not cfg:
        return False
    token = _cookies().get(cfg["cookie_name"])
    if token and _token_valid(token, cfg["cookie_secret"]):
        st.session_state["_is_editor"] = True
        return True
    return False


def try_login(pin: str) -> bool:
    """Verifica el PIN; si es correcto setea la cookie y la sesion."""
    cfg = auth_cfg()
    if not cfg:
        return False
    if not hmac.compare_digest(hash_pin(pin.strip()), cfg["pin_hash"]):
        return False
    token = _make_token(cfg["cookie_secret"], cfg["cookie_days"])
    expires = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=cfg["cookie_days"])
    _cookies().set(cfg["cookie_name"], token, expires_at=expires)
    st.session_state["_is_editor"] = True
    return True


def logout() -> None:
    cfg = auth_cfg()
    if cfg:
        try:
            _cookies().delete(cfg["cookie_name"])
        except Exception:
            pass
    st.session_state["_is_editor"] = False
