"""Configuration helpers: environment variables, optional .env file and Streamlit secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_TITLE = "Virtual Yoga Trainer"
APP_SUBTITLE = "Real-time pose feedback with MoveNet - explainable, personalised, rule-based"


def load_env_file(path: Optional[Path] = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (existing variables win)."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def push_streamlit_secrets_to_env() -> None:
    """Copy Streamlit Cloud secrets into environment variables (safe when no secrets exist)."""
    try:
        import streamlit as st

        for key in ("DATABASE_URL", "MOVENET_VARIANT", "MOVENET_MODEL_URL", "TURN_URL",
                    "TURN_USERNAME", "TURN_CREDENTIAL"):
            if key not in os.environ and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no secrets file / not running in Streamlit
        pass


def get_setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def get_ice_servers() -> list:
    """STUN (always) plus an optional TURN server for networks that block peer-to-peer video."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn = get_setting("TURN_URL")
    if turn:
        entry = {"urls": [turn]}
        if get_setting("TURN_USERNAME"):
            entry["username"] = get_setting("TURN_USERNAME")
            entry["credential"] = get_setting("TURN_CREDENTIAL")
        servers.append(entry)
    return servers
