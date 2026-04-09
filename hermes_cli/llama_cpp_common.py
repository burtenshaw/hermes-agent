"""Shared constants and filesystem helpers for managed llama.cpp support."""

from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

LLAMA_CPP_PROVIDER = "llama-cpp"
LLAMA_CPP_PROVIDER_ALIASES = ("llama-cpp", "llama.cpp", "llamacpp", "local")
LLAMA_CPP_DEFAULT_PORT = 8081
LLAMA_CPP_DEFAULT_VERSION = os.getenv("HERMES_LLAMA_CPP_VERSION", "latest").strip() or "latest"
LLAMA_CPP_RELEASE_REPO = "ggml-org/llama.cpp"
LLAMA_CPP_STATE_VERSION = 1
_WINDOWS = platform.system() == "Windows"
_STATE_UNUSED_KEYS = (
    "command",
    "hardware_snapshot",
    "last_health_check",
    "parallel_tool_calls_validated",
    "requested_acceleration",
    "binary_acceleration",
)


class LlamaCppError(RuntimeError):
    """User-facing managed-runtime error."""


ProgressCallback = Optional[Callable[[str], None]]


def get_engine_root() -> Path:
    return get_hermes_home() / "local_engines" / "llama_cpp"


def get_engine_bin_dir() -> Path:
    return get_engine_root() / "bin"


def get_engine_logs_dir() -> Path:
    return get_engine_root() / "logs"


def get_engine_state_path() -> Path:
    return get_engine_root() / "state.json"


def get_engine_binary_name() -> str:
    return "llama-server.exe" if _WINDOWS else "llama-server"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mkdirs() -> None:
    get_engine_bin_dir().mkdir(parents=True, exist_ok=True)
    get_engine_logs_dir().mkdir(parents=True, exist_ok=True)


def get_llama_cpp_cache_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "llama.cpp"
    if _WINDOWS:
        local_appdata = os.getenv("LOCALAPPDATA", "").strip()
        if local_appdata:
            return Path(local_appdata) / "llama.cpp"
        return Path.home() / "AppData" / "Local" / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


def load_state() -> Dict[str, Any]:
    path = get_engine_state_path()
    if not path.exists():
        return {"state_version": LLAMA_CPP_STATE_VERSION}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("state_version", LLAMA_CPP_STATE_VERSION)
            return data
    except Exception as exc:
        logger.debug("Failed to read llama.cpp state: %s", exc)
    return {"state_version": LLAMA_CPP_STATE_VERSION}


def save_state(state: Dict[str, Any]) -> None:
    _mkdirs()
    payload = dict(state or {})
    payload["state_version"] = LLAMA_CPP_STATE_VERSION
    for key in _STATE_UNUSED_KEYS:
        payload.pop(key, None)
    atomic_json_write(get_engine_state_path(), payload)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged
