"""Managed llama.cpp runtime support for Hermes.

This module owns the dedicated ``llama-cpp`` provider:

- curated model selection
- managed ``llama-server`` lifecycle
- persisted runtime state under ``HERMES_HOME``
- smoke tests for tool-calling readiness

The implementation is intentionally conservative. Hermes only treats
``llama-cpp`` as a first-class engine when the runtime is known-good and the
selected model comes from the curated allowlist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import requests

from hermes_constants import display_hermes_home, get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

LLAMA_CPP_PROVIDER = "llama-cpp"
LLAMA_CPP_PROVIDER_ALIASES = ("llama-cpp", "llama.cpp", "llamacpp", "local")
LLAMA_CPP_DEFAULT_PORT = 8081
LLAMA_CPP_DEFAULT_VERSION = os.getenv("HERMES_LLAMA_CPP_VERSION", "latest").strip() or "latest"
LLAMA_CPP_RELEASE_REPO = "ggml-org/llama.cpp"
LLAMA_CPP_STATE_VERSION = 1
_DEFAULT_CONTEXT_LENGTH = 32768
_SMOKE_TIMEOUT_SECONDS = 45.0
_DEFAULT_PARSER_CHAIN = ["qwen3_coder", "qwen", "llama3_json", "hermes"]
_PROGRESS_POLL_INTERVAL_SECONDS = 1.0

CURATED_MODELS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "tier": "tiny",
        "model_repo": "unsloth/Qwen3.5-2B-GGUF",
        "quant": "UD-Q4_K_XL",
        "context_length": 32768,
        "template_strategy": "native",
        "parser_chain": list(_DEFAULT_PARSER_CHAIN),
    },
    "balanced": {
        "tier": "balanced",
        "model_repo": "unsloth/Qwen3.5-9B-GGUF",
        "quant": "UD-Q4_K_XL",
        "context_length": 32768,
        "template_strategy": "native",
        "parser_chain": list(_DEFAULT_PARSER_CHAIN),
    },
    "large": {
        "tier": "large",
        "model_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "quant": "UD-Q4_K_XL",
        "context_length": 32768,
        "template_strategy": "native",
        "parser_chain": list(_DEFAULT_PARSER_CHAIN),
    },
}

_CURATED_SPECS = {
    f"{entry['model_repo']}:{entry['quant']}": dict(entry)
    for entry in CURATED_MODELS.values()
}

_WINDOWS = platform.system() == "Windows"


class LlamaCppError(RuntimeError):
    """User-facing managed-runtime error."""


ProgressCallback = Optional[Callable[[str], None]]


def default_engine_config() -> Dict[str, Any]:
    return {
        "managed": True,
        "auto_start": True,
        "port": LLAMA_CPP_DEFAULT_PORT,
        "selected_tier": "",
        "model_repo": "",
        "quant": "",
        "context_length": _DEFAULT_CONTEXT_LENGTH,
        "reasoning_budget": 0,
        "template_strategy": "native",
        "template_file": "",
        "parallel_tool_calls": False,
        "streaming_tool_calls": True,
    }


def is_llama_cpp_provider(value: Optional[str]) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in LLAMA_CPP_PROVIDER_ALIASES


def normalize_provider_name(value: Optional[str]) -> str:
    return LLAMA_CPP_PROVIDER if is_llama_cpp_provider(value) else str(value or "").strip().lower()


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
    payload.pop("hardware_snapshot", None)
    atomic_json_write(get_engine_state_path(), payload)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_engine_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    engine_cfg: Dict[str, Any] = {}
    if isinstance(config, dict):
        if isinstance(config.get("local_engines"), dict):
            local_engines = config.get("local_engines") or {}
            candidate = local_engines.get("llama_cpp")
            if isinstance(candidate, dict):
                engine_cfg = candidate
        elif any(key in config for key in default_engine_config().keys()):
            engine_cfg = config
    merged = _merge_dict(default_engine_config(), engine_cfg)
    try:
        merged["port"] = int(merged.get("port") or LLAMA_CPP_DEFAULT_PORT)
    except Exception:
        merged["port"] = LLAMA_CPP_DEFAULT_PORT
    try:
        merged["context_length"] = int(merged.get("context_length") or _DEFAULT_CONTEXT_LENGTH)
    except Exception:
        merged["context_length"] = _DEFAULT_CONTEXT_LENGTH
    try:
        merged["reasoning_budget"] = int(merged.get("reasoning_budget"))
    except Exception:
        merged["reasoning_budget"] = 0
    merged["selected_tier"] = str(merged.get("selected_tier") or "").strip().lower()
    merged["template_strategy"] = str(merged.get("template_strategy") or "native").strip().lower()
    merged["template_file"] = str(merged.get("template_file") or "").strip()
    merged["model_repo"] = str(merged.get("model_repo") or "").strip()
    merged["quant"] = str(merged.get("quant") or "").strip()
    merged["managed"] = bool(merged.get("managed", True))
    merged["auto_start"] = bool(merged.get("auto_start", True))
    merged["parallel_tool_calls"] = bool(merged.get("parallel_tool_calls", False))
    merged["streaming_tool_calls"] = bool(merged.get("streaming_tool_calls", False))
    return merged


def runtime_base_url(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = get_engine_config(config)
    return f"http://127.0.0.1:{cfg['port']}/v1"


def spec_string(model_repo: str, quant: str) -> str:
    repo = str(model_repo or "").strip()
    q = str(quant or "").strip()
    return f"{repo}:{q}" if repo and q else repo


def parse_model_spec(value: Optional[str]) -> Dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {"model_repo": "", "quant": ""}
    if ":" not in text:
        return {"model_repo": text, "quant": ""}
    repo, quant = text.rsplit(":", 1)
    return {"model_repo": repo.strip(), "quant": quant.strip()}


def curated_model_specs() -> list[str]:
    return list(_CURATED_SPECS.keys())


def _emit_progress(progress_callback: ProgressCallback, message: str) -> None:
    if not progress_callback:
        return
    text = str(message or "").strip()
    if not text:
        return
    try:
        progress_callback(text)
    except Exception:
        logger.debug("llama.cpp progress callback failed", exc_info=True)


def _format_bytes(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes or 0)))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            if size >= 100:
                return f"{size:.0f} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "0 B"


def _cache_repo_fragment(model_repo: str, separator: str) -> str:
    return str(model_repo or "").strip().replace("/", separator)


def _cache_manifest_path(model_repo: str, quant: str) -> Path:
    cache_dir = get_llama_cpp_cache_dir()
    return cache_dir / f"manifest={_cache_repo_fragment(model_repo, '=')}={quant}.json"


def _cache_artifact_paths(model_repo: str, filename: str) -> tuple[Path, Path]:
    cache_dir = get_llama_cpp_cache_dir()
    prefix = _cache_repo_fragment(model_repo, "_")
    final_path = cache_dir / f"{prefix}_{filename}"
    partial_path = Path(str(final_path) + ".downloadInProgress")
    return final_path, partial_path


def _read_cache_manifest(entry: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = _cache_manifest_path(entry.get("model_repo", ""), entry.get("quant", ""))
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def describe_startup_progress(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    entry = selected_model_entry(cfg)
    manifest = _read_cache_manifest(entry)
    if not manifest:
        return {}

    artifacts = (
        ("ggufFile", "model"),
        ("mmprojFile", "projector"),
    )
    for key, label in artifacts:
        artifact = manifest.get(key)
        if not isinstance(artifact, dict):
            continue
        filename = str(artifact.get("rfilename") or "").strip()
        total_bytes = artifact.get("size")
        if not isinstance(total_bytes, int) or total_bytes <= 0:
            lfs = artifact.get("lfs")
            if isinstance(lfs, dict):
                total_bytes = lfs.get("size")
        try:
            total_bytes = int(total_bytes or 0)
        except Exception:
            total_bytes = 0
        if not filename or total_bytes <= 0:
            continue

        final_path, partial_path = _cache_artifact_paths(entry["model_repo"], filename)
        if partial_path.exists():
            current_bytes = partial_path.stat().st_size
            percent = max(0, min(100, round((current_bytes / total_bytes) * 100)))
            return {
                "phase": "downloading",
                "artifact": label,
                "current_bytes": current_bytes,
                "total_bytes": total_bytes,
                "message": (
                    f"Downloading {label} {_format_bytes(current_bytes)} / "
                    f"{_format_bytes(total_bytes)} ({percent}%)"
                ),
            }

        if final_path.exists():
            current_bytes = final_path.stat().st_size
            if current_bytes < total_bytes:
                percent = max(0, min(100, round((current_bytes / total_bytes) * 100)))
                return {
                    "phase": "downloading",
                    "artifact": label,
                    "current_bytes": current_bytes,
                    "total_bytes": total_bytes,
                    "message": (
                        f"Downloading {label} {_format_bytes(current_bytes)} / "
                        f"{_format_bytes(total_bytes)} ({percent}%)"
                    ),
                }
            continue

        return {
            "phase": "preparing",
            "artifact": label,
            "message": f"Preparing {label} download...",
        }

    return {
        "phase": "loading",
        "message": "Loading model into memory...",
    }


def curated_entry_for_tier(tier: str) -> Dict[str, Any]:
    normalized = str(tier or "").strip().lower()
    if normalized not in CURATED_MODELS:
        raise LlamaCppError(f"Unknown llama.cpp tier '{tier}'.")
    return dict(CURATED_MODELS[normalized])


def curated_entry_for_spec(model_repo: str, quant: str) -> Optional[Dict[str, Any]]:
    return _CURATED_SPECS.get(spec_string(model_repo, quant))


def recommended_parser_chain(config: Optional[Dict[str, Any]] = None) -> list[str]:
    cfg = get_engine_config(config)
    current = curated_entry_for_spec(cfg.get("model_repo", ""), cfg.get("quant", ""))
    if current:
        return list(current.get("parser_chain") or _DEFAULT_PARSER_CHAIN)
    tier = cfg.get("selected_tier") or "balanced"
    if tier not in CURATED_MODELS:
        tier = "balanced"
    return list(curated_entry_for_tier(tier).get("parser_chain") or _DEFAULT_PARSER_CHAIN)


def effective_reasoning_budget(config: Optional[Dict[str, Any]] = None) -> int:
    cfg = get_engine_config(config)
    try:
        return int(cfg.get("reasoning_budget", 0))
    except Exception:
        return 0


def binary_candidates() -> list[Path]:
    state = load_state()
    candidates = []
    env_binary = os.getenv("HERMES_LLAMA_CPP_BINARY", "").strip()
    if env_binary:
        candidates.append(Path(env_binary))
    path_binary = shutil.which(get_engine_binary_name()) or shutil.which("llama-server")
    if path_binary:
        candidates.append(Path(path_binary))
    state_binary = str(state.get("binary_path") or "").strip()
    if state_binary:
        candidates.append(Path(state_binary))
    managed_binary = get_engine_bin_dir() / get_engine_binary_name()
    candidates.append(managed_binary)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_binary_path() -> Optional[Path]:
    for candidate in binary_candidates():
        if candidate.exists():
            return candidate
    return None


def read_binary_version(binary_path: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        text = (result.stdout or result.stderr or "").strip()
        if text:
            return text.splitlines()[0].strip()
    except Exception:
        pass
    return ""


def _download_file(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent/llama-cpp"})
    with urllib.request.urlopen(req, timeout=60) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_release_target() -> tuple[str, list[str]]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    target = f"{system}-{machine}"

    patterns: list[str]
    if system == "darwin":
        if machine in {"arm64", "aarch64"}:
            patterns = ["macos-arm64", "darwin-arm64", "metal", "arm64", "apple"]
        else:
            patterns = ["macos-x64", "darwin-x64", "x86_64", "amd64"]
    elif system == "linux":
        if machine in {"arm64", "aarch64"}:
            patterns = ["linux-arm64", "linux-aarch64", "arm64", "aarch64"]
        else:
            patterns = ["linux-x64", "linux-amd64", "x86_64", "amd64"]
    elif system == "windows":
        patterns = ["win64", "windows-x64", "windows-amd64", "x64", "amd64"]
    else:
        raise LlamaCppError(f"Automatic llama.cpp install is not supported on {platform.system()}/{platform.machine()}.")

    return target, patterns


def _release_api_url(version: str) -> str:
    if version == "latest":
        return f"https://api.github.com/repos/{LLAMA_CPP_RELEASE_REPO}/releases/latest"
    return f"https://api.github.com/repos/{LLAMA_CPP_RELEASE_REPO}/releases/tags/{version}"


def _fetch_release_metadata(version: str) -> Dict[str, Any]:
    req = urllib.request.Request(_release_api_url(version), headers={"User-Agent": "hermes-agent/llama-cpp"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LlamaCppError(f"Could not fetch llama.cpp release metadata ({exc.code}).") from exc
    except Exception as exc:
        raise LlamaCppError(f"Could not fetch llama.cpp release metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise LlamaCppError("llama.cpp release metadata was malformed.")
    return payload


def _score_asset(name: str, patterns: Iterable[str], prefer_cuda: bool) -> int:
    lowered = name.lower()
    score = 0
    for pattern in patterns:
        if pattern in lowered:
            score += 25
    if "server" in lowered or "bin" in lowered:
        score += 10
    if lowered.endswith((".zip", ".tar.gz", ".tgz")):
        score += 5
    if "cuda" in lowered:
        score += 10 if prefer_cuda else -5
    if "metal" in lowered and platform.system() == "Darwin":
        score += 5
    return score


def _resolve_release_asset(version: str) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    metadata = _fetch_release_metadata(version)
    assets = metadata.get("assets")
    if not isinstance(assets, list) or not assets:
        raise LlamaCppError("No downloadable assets found in the selected llama.cpp release.")

    _, patterns = _detect_release_target()
    prefer_cuda = False
    archive_asset: Optional[Dict[str, Any]] = None
    best_score = -1
    checksum_asset: Optional[Dict[str, Any]] = None

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        lowered = name.lower()
        if lowered in {"checksums.txt", "sha256sum.txt", "sha256sums.txt"} or lowered.endswith("checksums.txt"):
            checksum_asset = asset
            continue
        score = _score_asset(name, patterns, prefer_cuda)
        if score > best_score:
            best_score = score
            archive_asset = asset

    if archive_asset is None or best_score <= 0:
        raise LlamaCppError(
            "Could not find a compatible llama.cpp release asset for this platform. "
            "Set HERMES_LLAMA_CPP_BINARY to an existing llama-server binary to continue."
        )
    return archive_asset, checksum_asset


def _verify_checksum(archive_path: Path, checksum_path: Path, archive_name: str) -> bool:
    try:
        expected = None
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            parts = re.split(r"\s+", line.strip(), maxsplit=1)
            if len(parts) == 2 and parts[1].strip().lstrip("*") == archive_name:
                expected = parts[0].strip()
                break
        if not expected:
            return False
        return _sha256_of(archive_path) == expected
    except Exception:
        return False


def _install_release_binary(version: str) -> Path:
    archive_asset, checksum_asset = _resolve_release_asset(version)
    archive_url = str(archive_asset.get("browser_download_url") or "").strip()
    archive_name = str(archive_asset.get("name") or "").strip()
    if not archive_url or not archive_name:
        raise LlamaCppError("llama.cpp release metadata did not include a usable download URL.")

    _mkdirs()
    with tempfile.TemporaryDirectory(prefix="hermes-llama-cpp-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        archive_path = tmpdir / archive_name
        _download_file(archive_url, archive_path)

        if checksum_asset and checksum_asset.get("browser_download_url"):
            checksum_path = tmpdir / str(checksum_asset.get("name") or "checksums.txt")
            try:
                _download_file(str(checksum_asset["browser_download_url"]), checksum_path)
                if not _verify_checksum(archive_path, checksum_path, archive_name):
                    raise LlamaCppError("Downloaded llama.cpp archive failed checksum verification.")
            except LlamaCppError:
                raise
            except Exception as exc:
                logger.info("llama.cpp checksum verification unavailable: %s", exc)

        extract_dir = tmpdir / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        lowered = archive_name.lower()
        if lowered.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extract_dir)
        elif lowered.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extract_dir)
        else:
            raise LlamaCppError(f"Unsupported llama.cpp archive format: {archive_name}")

        binary_name = get_engine_binary_name()
        source_binary = None
        for candidate in extract_dir.rglob(binary_name):
            if candidate.is_file():
                source_binary = candidate
                break
        if source_binary is None:
            raise LlamaCppError("Downloaded llama.cpp archive did not contain llama-server.")

        destination = get_engine_bin_dir() / binary_name
        shutil.move(str(source_binary), str(destination))
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return destination


def ensure_binary_installed(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = get_engine_config(config)
    state = load_state()

    binary_path = resolve_binary_path()
    if binary_path is not None:
        state["binary_path"] = str(binary_path)
        version = read_binary_version(binary_path)
        if version:
            state["installed_version"] = version
        state["binary_checked_at"] = _utc_now_iso()
        save_state(state)
        return binary_path

    if not cfg.get("managed", True):
        raise LlamaCppError(
            "Managed llama.cpp is disabled and no llama-server binary was found. "
            "Set HERMES_LLAMA_CPP_BINARY or enable local_engines.llama_cpp.managed."
        )

    binary_path = _install_release_binary(LLAMA_CPP_DEFAULT_VERSION)
    state["binary_path"] = str(binary_path)
    state["installed_version"] = read_binary_version(binary_path)
    state["binary_checked_at"] = _utc_now_iso()
    save_state(state)
    return binary_path


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _probe_server(base_url: str) -> Dict[str, Any]:
    result = {
        "healthy": False,
        "models": [],
        "actual_model_id": "",
        "props": None,
    }
    headers = {"Authorization": "Bearer no-key-required"}
    try:
        models_resp = requests.get(base_url.rstrip("/") + "/models", headers=headers, timeout=5)
        if not models_resp.ok:
            return result
        payload = models_resp.json()
        models = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        result["models"] = models
        if len(models) == 1:
            result["actual_model_id"] = models[0]

        props_resp = requests.get(base_url.rstrip("/").replace("/v1", "") + "/v1/props", headers=headers, timeout=5)
        if not props_resp.ok:
            props_resp = requests.get(base_url.rstrip("/").replace("/v1", "") + "/props", headers=headers, timeout=5)
        if props_resp.ok:
            props = props_resp.json()
            result["props"] = props
            alias = props.get("model_alias")
            if alias and not result["actual_model_id"]:
                result["actual_model_id"] = str(alias)
        result["healthy"] = True
    except Exception:
        pass
    return result


def _terminate_existing_process(state: Dict[str, Any]) -> None:
    pid = state.get("pid")
    if not _pid_is_alive(pid):
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            break
        time.sleep(0.25)
    if _pid_is_alive(pid):
        try:
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass


def _kill_server_on_port(port: int) -> None:
    """Find and kill any llama-server process listening on *port*."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return
    for pid_str in out.splitlines():
        try:
            pid = int(pid_str.strip())
            os.kill(pid, signal.SIGTERM)
        except Exception:
            continue
    # Wait briefly for the port to free up.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if not out:
                return
        except Exception:
            return
        time.sleep(0.5)


def selected_model_entry(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    model_repo = cfg.get("model_repo", "").strip()
    quant = cfg.get("quant", "").strip()
    # Check curated list first.
    current = curated_entry_for_spec(model_repo, quant)
    if current is not None:
        merged = dict(current)
        merged["context_length"] = cfg.get("context_length") or current.get("context_length") or _DEFAULT_CONTEXT_LENGTH
        return merged
    # Explicit repo+quant not in the curated list — use as-is (custom model).
    if model_repo:
        return {
            "tier": cfg.get("selected_tier") or "",
            "model_repo": model_repo,
            "quant": quant,
            "context_length": cfg.get("context_length") or _DEFAULT_CONTEXT_LENGTH,
            "template_strategy": cfg.get("template_strategy", "native"),
            "parser_chain": list(_DEFAULT_PARSER_CHAIN),
        }
    # Nothing configured — fall back to a curated tier.
    tier = cfg.get("selected_tier") or "balanced"
    if tier not in CURATED_MODELS:
        tier = "balanced"
    selected = curated_entry_for_tier(tier)
    selected["context_length"] = cfg.get("context_length") or selected.get("context_length") or _DEFAULT_CONTEXT_LENGTH
    return selected


def build_server_command(
    *,
    binary_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> list[str]:
    cfg = get_engine_config(config)
    entry = selected_model_entry(cfg)
    model_spec = spec_string(entry["model_repo"], entry["quant"])
    command = [str(binary_path), "-hf", model_spec]
    command.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(cfg["port"]),
            "--jinja",
            "--reasoning-budget",
            str(effective_reasoning_budget(config)),
            "-c",
            str(entry.get("context_length") or _DEFAULT_CONTEXT_LENGTH),
        ]
    )
    template_strategy = cfg.get("template_strategy", "native")
    template_file = str(cfg.get("template_file") or "").strip()
    if template_strategy == "override" and template_file:
        command.extend(["--chat-template-file", template_file])
    return command


def start_server(
    config: Optional[Dict[str, Any]] = None,
    *,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    state = load_state()
    base_url = runtime_base_url(cfg)
    entry = selected_model_entry(cfg)
    model_spec = spec_string(entry["model_repo"], entry["quant"])
    last_progress_message = ""

    def report(message: str) -> None:
        nonlocal last_progress_message
        text = str(message or "").strip()
        if not text or text == last_progress_message:
            return
        last_progress_message = text
        _emit_progress(progress_callback, text)

    report(f"Checking local model {model_spec}...")
    probe = _probe_server(base_url)
    if probe["healthy"]:
        # Verify the running server has the correct model loaded.
        # A stale server from a previous config may still be listening
        # on the same port with a different (possibly broken) model.
        running_model = probe.get("actual_model_id") or ""
        if running_model == model_spec:
            state["actual_model_id"] = running_model
            state["last_health_check"] = _utc_now_iso()
            save_state(state)
            return get_status(cfg)
        report(f"Wrong model loaded ({running_model}), restarting...")
        # Kill whatever is on the port — may be a manually started server
        # or one from a previous config whose PID we no longer track.
        _kill_server_on_port(cfg["port"])

    if state.get("pid") and _pid_is_alive(state.get("pid")):
        _terminate_existing_process(state)

    report("Ensuring llama.cpp binary is ready...")
    binary_path = ensure_binary_installed(cfg)
    report("Launching llama-server...")
    command = build_server_command(binary_path=binary_path, config=cfg)
    log_path = get_engine_logs_dir() / f"llama-server-{int(time.time())}.log"
    _mkdirs()
    with open(log_path, "ab") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(get_engine_root()),
        )

    state.update(
        {
            "pid": process.pid,
            "binary_path": str(binary_path),
            "command": command,
            "log_path": str(log_path),
            "base_url": base_url,
            "started_at": _utc_now_iso(),
            "desired_model_spec": model_spec,
        }
    )
    save_state(state)

    deadline = time.time() + 180
    while time.time() < deadline:
        if process.poll() is not None:
            detail = f" Last step: {last_progress_message}." if last_progress_message else ""
            raise LlamaCppError(
                "Managed llama.cpp server exited early."
                f"{detail} Check {display_hermes_home()}/local_engines/llama_cpp/logs for details."
            )
        probe = _probe_server(base_url)
        if probe["healthy"]:
            state["actual_model_id"] = probe.get("actual_model_id") or ""
            state["last_health_check"] = _utc_now_iso()
            state["props"] = probe.get("props")
            save_state(state)
            return get_status(cfg)
        progress = describe_startup_progress(cfg)
        if progress.get("message"):
            report(str(progress["message"]))
        time.sleep(_PROGRESS_POLL_INTERVAL_SECONDS)

    detail = f" Last step: {last_progress_message}." if last_progress_message else ""
    raise LlamaCppError(
        f"Timed out waiting for llama.cpp to become healthy at {base_url}. "
        f"{detail} Check {display_hermes_home()}/local_engines/llama_cpp/logs."
    )


def stop_server() -> None:
    state = load_state()
    _terminate_existing_process(state)
    state["pid"] = None
    state["stopped_at"] = _utc_now_iso()
    save_state(state)


def _chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    max_tokens: int = 128,
    parallel_tool_calls: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = parallel_tool_calls
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": "Bearer no-key-required",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=_SMOKE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}


def _extract_tool_calls_from_response(response: Dict[str, Any]) -> list[Dict[str, Any]]:
    choices = response.get("choices") or []
    if not choices:
        return []
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list):
        return [tc for tc in tool_calls if isinstance(tc, dict)]

    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return []
    try:
        from environments.tool_call_parsers import get_parser
    except Exception:
        return []

    for parser_name in _DEFAULT_PARSER_CHAIN:
        try:
            _, parsed = get_parser(parser_name).parse(content)
        except Exception:
            continue
        if parsed:
            extracted = []
            for item in parsed:
                extracted.append(
                    {
                        "id": getattr(item, "id", ""),
                        "type": getattr(item, "type", "function"),
                        "function": {
                            "name": getattr(getattr(item, "function", None), "name", ""),
                            "arguments": getattr(getattr(item, "function", None), "arguments", "{}"),
                        },
                    }
                )
            return extracted
    return []


def run_smoke_tests(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    status = get_status(cfg)
    if not status.get("healthy"):
        raise LlamaCppError("Cannot run llama.cpp smoke tests because the managed server is not healthy.")

    model = status.get("actual_model_id") or status.get("model") or selected_model_entry(cfg)["model_repo"]
    base_url = runtime_base_url(cfg)
    results: Dict[str, Any] = {
        "checked_at": _utc_now_iso(),
        "props_verified": False,
        "single_tool_call": False,
        "tool_followup": False,
        "malformed_recovery": False,
        "parallel_tool_calls": False,
        "passed": False,
        "errors": [],
    }

    props = status.get("props")
    if isinstance(props, dict) and props.get("default_generation_settings") is not None:
        results["props_verified"] = True
    else:
        results["errors"].append("llama.cpp /v1/props did not return default_generation_settings")

    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": "smoke_ping",
                "description": "Returns the provided integer value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                    },
                    "required": ["value"],
                },
            },
        }
    ]

    try:
        single = _chat_completion(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": "Use tools exactly when appropriate."},
                {"role": "user", "content": "Call smoke_ping with value 42. Do not answer in plain text first."},
            ],
            tools=tool_schema,
            max_tokens=128,
            parallel_tool_calls=False,
        )
        single_calls = _extract_tool_calls_from_response(single)
        if single_calls:
            first_call = single_calls[0]
            fn = first_call.get("function") if isinstance(first_call, dict) else {}
            if isinstance(fn, dict) and fn.get("name") == "smoke_ping":
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                if args.get("value") == 42:
                    results["single_tool_call"] = True
        if not results["single_tool_call"]:
            results["errors"].append("single tool-call smoke test did not produce the expected smoke_ping(42) call")
    except Exception as exc:
        results["errors"].append(f"single tool-call smoke test failed: {exc}")

    try:
        followup = _chat_completion(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": "Keep responses short."},
                {"role": "user", "content": "Call smoke_ping with value 7."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_smoke_ping",
                            "type": "function",
                            "function": {"name": "smoke_ping", "arguments": '{"value": 7}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_smoke_ping", "content": '{"result": 7}'},
                {"role": "user", "content": "State the result you received from the tool."},
            ],
            max_tokens=64,
        )
        choices = followup.get("choices") or []
        if choices:
            content = (((choices[0] or {}).get("message") or {}).get("content") or "").lower()
            if "7" in content:
                results["tool_followup"] = True
        if not results["tool_followup"]:
            results["errors"].append("tool follow-up smoke test did not reflect the tool result")
    except Exception as exc:
        results["errors"].append(f"tool follow-up smoke test failed: {exc}")

    try:
        malformed = _chat_completion(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": "Retry tool calls when the previous attempt had invalid JSON."},
                {"role": "user", "content": "Call smoke_ping with value 13."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bad_json",
                            "type": "function",
                            "function": {"name": "smoke_ping", "arguments": '{"value": }'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bad_json",
                    "content": "Error: Invalid JSON arguments. Please retry with valid JSON.",
                },
                {"role": "user", "content": "Retry the tool call with valid JSON only."},
            ],
            tools=tool_schema,
            max_tokens=128,
        )
        malformed_calls = _extract_tool_calls_from_response(malformed)
        if malformed_calls:
            fn = malformed_calls[0].get("function") if isinstance(malformed_calls[0], dict) else {}
            if isinstance(fn, dict):
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                if fn.get("name") == "smoke_ping" and args.get("value") == 13:
                    results["malformed_recovery"] = True
        if not results["malformed_recovery"]:
            results["errors"].append("malformed-argument recovery smoke test did not produce a corrected tool call")
    except Exception as exc:
        results["errors"].append(f"malformed-argument recovery smoke test failed: {exc}")

    try:
        parallel = _chat_completion(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": "If the model supports parallel tool calls, emit two smoke_ping calls in one response."},
                {"role": "user", "content": "Call smoke_ping twice, once with 1 and once with 2, in the same response if supported."},
            ],
            tools=tool_schema,
            max_tokens=128,
            parallel_tool_calls=True,
        )
        parallel_calls = _extract_tool_calls_from_response(parallel)
        seen = set()
        for item in parallel_calls:
            fn = item.get("function") if isinstance(item, dict) else {}
            if not isinstance(fn, dict):
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if fn.get("name") == "smoke_ping" and args.get("value") in {1, 2}:
                seen.add(args.get("value"))
        results["parallel_tool_calls"] = seen == {1, 2}
    except Exception:
        results["parallel_tool_calls"] = False

    results["passed"] = all(
        results.get(key) for key in ("props_verified", "single_tool_call", "tool_followup", "malformed_recovery")
    )
    state = load_state()
    state["smoke_tests"] = results
    if results["parallel_tool_calls"]:
        state["parallel_tool_calls_validated"] = True
    save_state(state)
    return results


def get_status(config: Optional[Dict[str, Any]] = None, *, check_health: bool = True) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    state = load_state()
    binary_path = resolve_binary_path()
    base_url = runtime_base_url(cfg)
    entry = selected_model_entry(cfg)

    status = {
        "provider": LLAMA_CPP_PROVIDER,
        "managed": bool(cfg.get("managed", True)),
        "configured": True,
        "auto_start": bool(cfg.get("auto_start", True)),
        "port": cfg["port"],
        "base_url": base_url,
        "binary_path": str(binary_path) if binary_path else str(state.get("binary_path") or ""),
        "installed": bool(binary_path),
        "installed_version": str(state.get("installed_version") or ""),
        "pid": state.get("pid"),
        "process_running": _pid_is_alive(state.get("pid")),
        "healthy": False,
        "model_repo": entry["model_repo"],
        "quant": entry["quant"],
        "model_spec": spec_string(entry["model_repo"], entry["quant"]),
        "selected_tier": entry.get("tier") or cfg.get("selected_tier") or "",
        "context_length": int(entry.get("context_length") or cfg.get("context_length") or _DEFAULT_CONTEXT_LENGTH),
        "reasoning_budget": effective_reasoning_budget(config),
        "template_strategy": cfg.get("template_strategy", "native"),
        "template_file": cfg.get("template_file") or "",
        "parallel_tool_calls": bool(cfg.get("parallel_tool_calls", False)),
        "streaming_tool_calls": bool(cfg.get("streaming_tool_calls", False)),
        "actual_model_id": str(state.get("actual_model_id") or ""),
        "log_path": str(state.get("log_path") or ""),
        "smoke_tests": state.get("smoke_tests") if isinstance(state.get("smoke_tests"), dict) else {},
        "props": state.get("props") if isinstance(state.get("props"), dict) else None,
    }

    if binary_path:
        version = read_binary_version(binary_path)
        if version:
            status["installed_version"] = version

    if check_health:
        probe = _probe_server(base_url)
        status["healthy"] = bool(probe.get("healthy"))
        if probe.get("actual_model_id"):
            status["actual_model_id"] = probe["actual_model_id"]
        if probe.get("props") is not None:
            status["props"] = probe["props"]
    return status


def ensure_runtime_ready(
    config: Optional[Dict[str, Any]] = None,
    *,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    _emit_progress(progress_callback, "Checking local llama.cpp runtime...")
    if not cfg.get("auto_start", True):
        status = get_status(cfg)
        if not status.get("healthy"):
            raise LlamaCppError(
                f"Managed llama.cpp is configured with auto_start=false and nothing is listening on {runtime_base_url(cfg)}."
            )
        return status

    status = get_status(cfg)
    desired_spec = spec_string(
        selected_model_entry(cfg)["model_repo"],
        selected_model_entry(cfg)["quant"],
    )
    running_model = status.get("actual_model_id") or ""
    if not status.get("healthy") or running_model != desired_spec:
        status = start_server(cfg, progress_callback=progress_callback)

    smoke = status.get("smoke_tests") if isinstance(status.get("smoke_tests"), dict) else {}
    if not smoke.get("passed"):
        _emit_progress(progress_callback, "Running llama.cpp smoke tests...")
        smoke = run_smoke_tests(cfg)
        status = get_status(cfg)
        status["smoke_tests"] = smoke
        if smoke.get("parallel_tool_calls"):
            status["parallel_tool_calls"] = True

    if not smoke.get("passed"):
        raise LlamaCppError(
            "Managed llama.cpp started, but Hermes' tool-calling smoke tests did not pass. "
            f"Check {display_hermes_home()}/local_engines/llama_cpp/logs and try a smaller model tier."
        )

    return status


def runtime_payload(
    config: Optional[Dict[str, Any]] = None,
    *,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    status = ensure_runtime_ready(config, progress_callback=progress_callback)
    actual_model = status.get("actual_model_id") or status.get("model_spec")
    return {
        "provider": LLAMA_CPP_PROVIDER,
        "api_mode": "chat_completions",
        "base_url": status["base_url"],
        "api_key": "no-key-required",
        "source": "managed:llama-cpp",
        "model": actual_model,
        "parallel_tool_calls": bool(status.get("parallel_tool_calls", False)),
        "streaming_tool_calls": bool(status.get("streaming_tool_calls", False)),
        "parser_chain": recommended_parser_chain(config),
    }


def sync_config_model_fields(config: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    runtime = runtime or get_status(cfg)
    entry = selected_model_entry(cfg)
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {"default": model_cfg} if model_cfg else {}
    model_cfg["provider"] = LLAMA_CPP_PROVIDER
    model_cfg["base_url"] = runtime_base_url(cfg)
    model_cfg["api_mode"] = "chat_completions"
    model_cfg["default"] = str(runtime.get("model_spec") or spec_string(entry["model_repo"], entry["quant"]))
    model_cfg["context_length"] = int(entry.get("context_length") or cfg.get("context_length") or _DEFAULT_CONTEXT_LENGTH)
    config["model"] = model_cfg
    return config


def configure_selected_model(
    config: Dict[str, Any],
    *,
    tier: str,
    model_repo: Optional[str] = None,
    quant: Optional[str] = None,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    local_engines = config.setdefault("local_engines", {})
    if not isinstance(local_engines, dict):
        local_engines = {}
        config["local_engines"] = local_engines
    engine_cfg = local_engines.setdefault("llama_cpp", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
        local_engines["llama_cpp"] = engine_cfg

    curated = curated_entry_for_tier(tier)
    engine_cfg["managed"] = True
    engine_cfg["auto_start"] = True
    engine_cfg["selected_tier"] = tier
    engine_cfg["model_repo"] = model_repo or curated["model_repo"]
    engine_cfg["quant"] = quant or curated["quant"]
    engine_cfg["context_length"] = int(context_length or curated.get("context_length") or _DEFAULT_CONTEXT_LENGTH)
    if "reasoning_budget" in engine_cfg:
        try:
            engine_cfg["reasoning_budget"] = int(engine_cfg.get("reasoning_budget"))
        except Exception:
            engine_cfg["reasoning_budget"] = 0
    else:
        engine_cfg["reasoning_budget"] = 0
    engine_cfg["template_strategy"] = curated.get("template_strategy", "native")
    return config


def get_managed_provider_status() -> Dict[str, Any]:
    try:
        status = get_status()
    except Exception as exc:
        return {
            "configured": False,
            "logged_in": False,
            "provider": LLAMA_CPP_PROVIDER,
            "name": "Local",
            "error": str(exc),
        }
    return {
        "configured": bool(status.get("installed") or status.get("healthy") or status.get("managed")),
        "logged_in": bool(status.get("healthy")),
        "provider": LLAMA_CPP_PROVIDER,
        "name": "Local",
        "base_url": status.get("base_url"),
        "binary_path": status.get("binary_path"),
        "healthy": bool(status.get("healthy")),
    }
