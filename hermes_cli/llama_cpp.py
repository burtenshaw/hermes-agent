"""Managed llama.cpp runtime support for Hermes.

This module owns the dedicated ``llama-cpp`` provider:

- curated model selection
- managed ``llama-server`` lifecycle
- persisted runtime state under ``HERMES_HOME``
- smoke tests for tool-calling readiness
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
import socket
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

from hermes_cli.llama_cpp_common import (
    LLAMA_CPP_DEFAULT_PORT,
    LLAMA_CPP_DEFAULT_VERSION,
    LLAMA_CPP_PROVIDER,
    LLAMA_CPP_RELEASE_REPO,
    LlamaCppError,
    ProgressCallback,
    _WINDOWS,
    _mkdirs,
    _utc_now_iso,
    get_engine_bin_dir,
    get_engine_binary_name,
    get_engine_logs_dir,
    get_engine_root,
    get_llama_cpp_cache_dir,
    load_state,
    save_state,
)
from hermes_cli.llama_cpp_config import (
    _DEFAULT_CONTEXT_LENGTH,
    _DEFAULT_PARSER_CHAIN,
    agent_runtime_settings,
    canonical_model_value,
    configure_selected_model,
    curated_entry_for_tier,
    curated_entry_for_spec,
    effective_gpu_layers,
    curated_model_specs,
    effective_reasoning_budget,
    ensure_engine_config_section,
    get_engine_config,
    is_llama_cpp_provider,
    normalize_acceleration,
    parse_model_spec,
    recommended_parser_chain,
    runtime_base_url,
    selected_model_entry,
    spec_string,
    sync_config_model_fields,
)
from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)

_SMOKE_TIMEOUT_SECONDS = 45.0
_PROGRESS_POLL_INTERVAL_SECONDS = 1.0
_server_start_lock = threading.Lock()
_SMOKE_PING_TOOL_SCHEMA = [
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


def _env_binary_path() -> Optional[Path]:
    env_binary = os.getenv("HERMES_LLAMA_CPP_BINARY", "").strip()
    if not env_binary:
        return None
    return Path(env_binary)


def _path_binary_path() -> Optional[Path]:
    candidate = shutil.which(get_engine_binary_name()) or shutil.which("llama-server")
    return Path(candidate) if candidate else None


def resolve_binary_path(
    config: Optional[Dict[str, Any]] = None,
    *,
    state: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    cfg = get_engine_config(config)
    state = load_state() if state is None else state

    env_binary = _env_binary_path()
    if env_binary and env_binary.exists():
        return env_binary

    managed_binary = get_engine_bin_dir() / get_engine_binary_name()
    if cfg.get("managed", True):
        return managed_binary if managed_binary.exists() else None

    state_binary = str(state.get("binary_path") or "").strip()
    if state_binary:
        state_path = Path(state_binary)
        if state_path.exists():
            return state_path

    path_binary = _path_binary_path()
    if path_binary and path_binary.exists():
        return path_binary

    return managed_binary if managed_binary.exists() else None


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
            patterns = ["macos-arm64", "darwin-arm64", "arm64", "aarch64", "apple"]
        else:
            patterns = ["macos-x64", "darwin-x64", "x64", "x86_64", "amd64"]
    elif system == "linux":
        if machine in {"arm64", "aarch64"}:
            patterns = ["ubuntu-arm64", "linux-arm64", "linux-aarch64", "arm64", "aarch64"]
        else:
            patterns = ["ubuntu-x64", "linux-x64", "linux-amd64", "x64", "x86_64", "amd64"]
    elif system == "windows":
        if machine in {"arm64", "aarch64"}:
            patterns = ["win-arm64", "windows-arm64", "arm64", "aarch64"]
        else:
            patterns = ["win-x64", "windows-x64", "windows-amd64", "x64", "amd64"]
    else:
        raise LlamaCppError(f"Automatic llama.cpp install is not supported on {platform.system()}/{platform.machine()}.")

    return target, patterns


def _cuda_runtime_available() -> bool:
    return bool(
        shutil.which("nvidia-smi")
        or os.getenv("CUDA_PATH", "").strip()
        or os.getenv("CUDA_HOME", "").strip()
    )


def resolve_acceleration(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = get_engine_config(config)
    requested = normalize_acceleration(cfg.get("acceleration"))
    if requested != "auto":
        return requested

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "metal"
    if system in {"linux", "windows"} and _cuda_runtime_available():
        return "cuda"
    return "cpu"


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


def _asset_matches_acceleration(name: str, acceleration: str) -> bool:
    lowered = name.lower()
    system = platform.system().lower()
    machine = platform.machine().lower()

    if acceleration == "cpu":
        return "cuda" not in lowered
    if acceleration == "metal":
        return system == "darwin" and machine in {"arm64", "aarch64"} and "macos-arm64" in lowered
    if acceleration == "cuda":
        return "cuda" in lowered
    return False


def _score_asset(name: str, patterns: Iterable[str], acceleration: str) -> int:
    lowered = name.lower()
    if not lowered.endswith((".zip", ".tar.gz", ".tgz")):
        return -1
    if not _asset_matches_acceleration(lowered, acceleration):
        return -1

    score = 0
    matched_patterns = 0
    for pattern in patterns:
        if pattern in lowered:
            matched_patterns += 1
            score += 25
    if matched_patterns == 0:
        return -1

    if "server" in lowered or "bin" in lowered:
        score += 10
    if lowered.startswith("llama-"):
        score += 5
    if lowered.startswith("cudart-"):
        score -= 5
    score += 5
    if acceleration == "cpu":
        if "cpu" in lowered:
            score += 10
        if "kleidiai" in lowered:
            score -= 3
    elif acceleration == "metal":
        score += 20
        if "kleidiai" in lowered:
            score -= 3
    elif acceleration == "cuda":
        score += 20
        if "cuda" in lowered:
            score += 10
    return score


def _resolve_release_asset(
    version: str,
    *,
    acceleration: str,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    metadata = _fetch_release_metadata(version)
    assets = metadata.get("assets")
    if not isinstance(assets, list) or not assets:
        raise LlamaCppError("No downloadable assets found in the selected llama.cpp release.")

    checksum_asset: Optional[Dict[str, Any]] = None

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        lowered = name.lower()
        if lowered in {"checksums.txt", "sha256sum.txt", "sha256sums.txt"} or lowered.endswith("checksums.txt"):
            checksum_asset = asset
    _, patterns = _detect_release_target()
    resolved = normalize_acceleration(acceleration)
    archive_asset: Optional[Dict[str, Any]] = None
    best_score = -1
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "")
        score = _score_asset(name, patterns, resolved)
        if score > best_score:
            best_score = score
            archive_asset = asset
    if archive_asset is not None and best_score > 0:
        return archive_asset, checksum_asset

    raise LlamaCppError(
        f"Managed llama.cpp does not publish a compatible '{resolved}' binary for "
        f"{platform.system()}/{platform.machine()}. Set HERMES_LLAMA_CPP_BINARY to a custom build, "
        "or choose a different acceleration."
    )


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


def _install_release_binary(
    version: str,
    *,
    acceleration: str,
) -> tuple[Path, str]:
    archive_asset, checksum_asset = _resolve_release_asset(version, acceleration=acceleration)
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
        return destination, archive_name


def _managed_binary_path() -> Path:
    return get_engine_bin_dir() / get_engine_binary_name()


def _is_managed_binary_path(path: Path) -> bool:
    try:
        return path.resolve() == _managed_binary_path().resolve()
    except Exception:
        return str(path) == str(_managed_binary_path())


def _managed_asset_matches_acceleration(asset_name: str, acceleration: str) -> bool:
    name = str(asset_name or "").strip().lower()
    if not name:
        return False
    return _asset_matches_acceleration(name, normalize_acceleration(acceleration))


def ensure_binary_installed(config: Optional[Dict[str, Any]] = None) -> tuple[Path, str]:
    cfg = get_engine_config(config)
    state = load_state()
    requested_acceleration = normalize_acceleration(cfg.get("acceleration"))
    resolved_acceleration = resolve_acceleration(cfg)

    binary_path = resolve_binary_path(cfg, state=state)
    if binary_path is not None:
        if (
            cfg.get("managed", True)
            and _is_managed_binary_path(binary_path)
            and not _managed_asset_matches_acceleration(state.get("binary_asset_name", ""), resolved_acceleration)
        ):
            binary_path = None
        else:
            state["binary_path"] = str(binary_path)
            state["binary_source"] = "managed" if _is_managed_binary_path(binary_path) else "external"
            state["binary_checked_at"] = _utc_now_iso()
            if not _is_managed_binary_path(binary_path):
                state.pop("binary_asset_name", None)
            version = read_binary_version(binary_path)
            if version:
                state["installed_version"] = version
            save_state(state)
            return binary_path, resolved_acceleration

    if not cfg.get("managed", True):
        raise LlamaCppError(
            "Managed llama.cpp is disabled and no llama-server binary was found. "
            "Set HERMES_LLAMA_CPP_BINARY or enable local_engines.llama_cpp.managed."
        )

    install_acceleration = resolved_acceleration
    try:
        binary_path, asset_name = _install_release_binary(
            LLAMA_CPP_DEFAULT_VERSION,
            acceleration=install_acceleration,
        )
    except LlamaCppError:
        if requested_acceleration != "auto" or resolved_acceleration == "cpu":
            raise
        install_acceleration = "cpu"
        binary_path, asset_name = _install_release_binary(
            LLAMA_CPP_DEFAULT_VERSION,
            acceleration=install_acceleration,
        )

    state["binary_path"] = str(binary_path)
    state["binary_source"] = "managed"
    state["binary_asset_name"] = asset_name
    state["binary_checked_at"] = _utc_now_iso()
    state["installed_version"] = read_binary_version(binary_path)
    save_state(state)
    return binary_path, install_acceleration


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


def _terminate_pid(pid: int, *, force: bool = False) -> None:
    if _WINDOWS:
        command = ["taskkill", "/PID", str(pid), "/T"]
        command.append("/F")
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)


def _find_pids_on_port(port: int) -> list[int]:
    try:
        port = int(port)
    except Exception:
        return []

    try:
        import psutil

        pids = sorted(
            {
                int(conn.pid)
                for conn in psutil.net_connections(kind="tcp")
                if conn.pid
                and conn.status == psutil.CONN_LISTEN
                and getattr(conn, "laddr", None)
                and getattr(conn.laddr, "port", None) == port
            }
        )
        if pids:
            return pids
    except Exception:
        pass

    if not _WINDOWS and shutil.which("lsof"):
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return []
        pids: list[int] = []
        for pid_str in out.splitlines():
            try:
                pids.append(int(pid_str.strip()))
            except Exception:
                continue
        return pids

    return []


def _port_is_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def _terminate_existing_process(state: Dict[str, Any], *, force: bool = False) -> bool:
    pid = state.get("pid")
    if not _pid_is_alive(pid):
        return False
    try:
        _terminate_pid(int(pid), force=force)
    except Exception:
        return False
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            return True
        time.sleep(0.25)
    if _pid_is_alive(pid):
        try:
            _terminate_pid(int(pid), force=True)
        except Exception:
            pass
    return not _pid_is_alive(pid)


def _kill_server_on_port(port: int, *, force: bool = False) -> list[int]:
    """Find and kill any llama-server process listening on *port*."""
    pids = _find_pids_on_port(port)
    for pid in pids:
        try:
            _terminate_pid(pid, force=force)
        except Exception:
            continue
    # Wait briefly for the port to free up.
    deadline = time.time() + 5
    while time.time() < deadline:
        if not _port_is_listening(port):
            return pids
        time.sleep(0.5)
    for pid in pids:
        try:
            _terminate_pid(pid, force=True)
        except Exception:
            continue
    return pids


def build_server_command(
    *,
    binary_path: Path,
    config: Optional[Dict[str, Any]] = None,
    resolved_acceleration: Optional[str] = None,
) -> list[str]:
    cfg = get_engine_config(config)
    entry = selected_model_entry(cfg)
    command = [str(binary_path), "-hf", entry["model_spec"]]
    command.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(cfg["port"]),
            "--jinja",
            "--reasoning-format",
            cfg.get("reasoning_format", "deepseek"),
            "--reasoning-budget",
            str(effective_reasoning_budget(cfg)),
            "-c",
            str(entry.get("context_length") or _DEFAULT_CONTEXT_LENGTH),
        ]
    )
    gpu_layers = effective_gpu_layers(cfg, resolved_acceleration=resolved_acceleration)
    if gpu_layers > 0:
        configured_gpu_layers = int(cfg.get("gpu_layers", -1))
        gpu_layers_arg = "all" if configured_gpu_layers < 0 else str(gpu_layers)
        command.extend(["--n-gpu-layers", gpu_layers_arg])
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
    with _server_start_lock:
        return _start_server_locked(config, progress_callback=progress_callback)


def _start_server_locked(
    config: Optional[Dict[str, Any]] = None,
    *,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    state = load_state()
    base_url = runtime_base_url(cfg)
    entry = selected_model_entry(cfg)
    model_spec = entry["model_spec"]
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
        running_model = probe.get("actual_model_id") or ""
        if running_model == model_spec:
            state["actual_model_id"] = running_model
            save_state(state)
            return get_status(cfg)
        report(f"Wrong model loaded ({running_model}), restarting...")
        # Kill whatever is on the port — may be a manually started server
        # or one from a previous config whose PID we no longer track.
        _kill_server_on_port(cfg["port"])

    if state.get("pid") and _pid_is_alive(state.get("pid")):
        _terminate_existing_process(state)

    report("Ensuring llama.cpp binary is ready...")
    binary_path, resolved_acceleration = ensure_binary_installed(cfg)
    report("Launching llama-server...")
    command = build_server_command(
        binary_path=binary_path,
        config=cfg,
        resolved_acceleration=resolved_acceleration,
    )
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
            "log_path": str(log_path),
            "base_url": base_url,
            "desired_model_spec": model_spec,
            "gpu_layers": effective_gpu_layers(cfg, resolved_acceleration=resolved_acceleration),
            "started_at": _utc_now_iso(),
            "stopped_at": None,
        }
    )
    save_state(state)

    stall_timeout = int(cfg.get("startup_stall_timeout_seconds") or 600)
    last_activity_at = time.time()
    last_activity_marker: Optional[tuple[Any, ...]] = None
    while True:
        if process.poll() is not None:
            detail = f" Last step: {last_progress_message}." if last_progress_message else ""
            raise LlamaCppError(
                "Managed llama.cpp server exited early."
                f"{detail} Check {display_hermes_home()}/local_engines/llama_cpp/logs for details."
            )
        probe = _probe_server(base_url)
        if probe["healthy"]:
            state["actual_model_id"] = probe.get("actual_model_id") or ""
            state["props"] = probe.get("props")
            state["stopped_at"] = None
            save_state(state)
            return get_status(cfg)
        progress = describe_startup_progress(cfg)
        log_size = 0
        try:
            if log_path.exists():
                log_size = log_path.stat().st_size
        except Exception:
            log_size = 0
        activity_marker = (
            progress.get("phase"),
            progress.get("artifact"),
            progress.get("current_bytes"),
            progress.get("total_bytes"),
            progress.get("message"),
            log_size,
        )
        if activity_marker != last_activity_marker:
            last_activity_marker = activity_marker
            last_activity_at = time.time()
        if progress.get("message"):
            report(str(progress["message"]))
        if time.time() - last_activity_at >= stall_timeout:
            detail = f" Last step: {last_progress_message}." if last_progress_message else ""
            raise LlamaCppError(
                f"No startup progress from llama.cpp for {stall_timeout} seconds at {base_url}."
                f"{detail} Check {display_hermes_home()}/local_engines/llama_cpp/logs."
            )
        time.sleep(_PROGRESS_POLL_INTERVAL_SECONDS)


def stop_server(
    config: Optional[Dict[str, Any]] = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    state = load_state()
    terminated_pid = int(state["pid"]) if _pid_is_alive(state.get("pid")) else None
    stopped_process = _terminate_existing_process(state, force=force)
    killed_pids = []
    if _port_is_listening(cfg["port"]):
        killed_pids = _kill_server_on_port(cfg["port"], force=force)
    state["pid"] = None
    state["actual_model_id"] = ""
    state["props"] = None
    state["stopped_at"] = _utc_now_iso()
    save_state(state)
    return {
        "stopped": bool(stopped_process or killed_pids or terminated_pid),
        "terminated_pid": terminated_pid,
        "killed_pids": killed_pids,
        "base_url": runtime_base_url(cfg),
        "port": cfg["port"],
        "force": bool(force),
    }


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


def _extract_tool_calls_from_response(
    response: Dict[str, Any],
    *,
    parser_chain: Optional[Iterable[str]] = None,
) -> list[Dict[str, Any]]:
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

    for parser_name in parser_chain or _DEFAULT_PARSER_CHAIN:
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

    entry = selected_model_entry(cfg)
    parser_chain = entry.get("parser_chain") or _DEFAULT_PARSER_CHAIN
    model = status.get("actual_model_id") or status.get("model_spec") or entry["model_repo"]
    base_url = runtime_base_url(cfg)
    results: Dict[str, Any] = {
        "checked_at": _utc_now_iso(),
        "props_verified": False,
        "single_tool_call": False,
        "tool_followup": False,
        "passed": False,
        "errors": [],
    }

    props = status.get("props")
    if isinstance(props, dict) and props.get("default_generation_settings") is not None:
        results["props_verified"] = True
    else:
        results["errors"].append("llama.cpp /v1/props did not return default_generation_settings")

    try:
        single = _chat_completion(
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": "Use tools exactly when appropriate."},
                {"role": "user", "content": "Call smoke_ping with value 42. Do not answer in plain text first."},
            ],
            tools=_SMOKE_PING_TOOL_SCHEMA,
            max_tokens=128,
            parallel_tool_calls=False,
        )
        single_calls = _extract_tool_calls_from_response(single, parser_chain=parser_chain)
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

    results["passed"] = all(
        results.get(key) for key in ("props_verified", "single_tool_call", "tool_followup")
    )
    state = load_state()
    state["smoke_tests"] = results
    save_state(state)
    return results


def get_status(config: Optional[Dict[str, Any]] = None, *, check_health: bool = True) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    state = load_state()
    binary_path = resolve_binary_path(cfg, state=state)
    base_url = runtime_base_url(cfg)
    entry = selected_model_entry(cfg)
    requested_acceleration = normalize_acceleration(cfg.get("acceleration"))
    resolved_acceleration = resolve_acceleration(cfg)
    configured_gpu_layers = int(cfg.get("gpu_layers", -1))
    runtime_gpu_layers = effective_gpu_layers(cfg, resolved_acceleration=resolved_acceleration)

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
        "binary_source": str(state.get("binary_source") or ""),
        "binary_asset_name": str(state.get("binary_asset_name") or ""),
        "binary_checked_at": str(state.get("binary_checked_at") or ""),
        "pid": state.get("pid"),
        "process_running": _pid_is_alive(state.get("pid")),
        "healthy": False,
        "model_repo": entry["model_repo"],
        "quant": entry["quant"],
        "model_spec": entry["model_spec"],
        "selected_tier": entry.get("tier") or cfg.get("selected_tier") or "",
        "context_length": int(entry.get("context_length") or cfg.get("context_length") or _DEFAULT_CONTEXT_LENGTH),
        "reasoning_budget": effective_reasoning_budget(config),
        "template_strategy": cfg.get("template_strategy", "native"),
        "template_file": cfg.get("template_file") or "",
        "parallel_tool_calls": bool(cfg.get("parallel_tool_calls", False)),
        "requested_acceleration": requested_acceleration,
        "resolved_acceleration": resolved_acceleration,
        "configured_gpu_layers": configured_gpu_layers,
        "gpu_layers": runtime_gpu_layers,
        "startup_stall_timeout_seconds": int(cfg.get("startup_stall_timeout_seconds") or 600),
        "actual_model_id": str(state.get("actual_model_id") or ""),
        "log_path": str(state.get("log_path") or ""),
        "started_at": str(state.get("started_at") or ""),
        "stopped_at": str(state.get("stopped_at") or ""),
        "desired_model_spec": str(state.get("desired_model_spec") or entry["model_spec"]),
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
        if status["healthy"]:
            state["actual_model_id"] = status["actual_model_id"]
            state["props"] = status["props"]
            state["stopped_at"] = None
            save_state(state)
    return status


def read_recent_logs(
    config: Optional[Dict[str, Any]] = None,
    *,
    lines: int = 80,
) -> str:
    del config
    state = load_state()
    candidates: list[Path] = []
    log_path = str(state.get("log_path") or "").strip()
    if log_path:
        candidates.append(Path(log_path))
    logs_dir = get_engine_logs_dir()
    if logs_dir.exists():
        try:
            recent = sorted(
                logs_dir.glob("llama-server-*.log"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            recent = []
        candidates.extend(recent)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        state["log_path"] = str(candidate)
        save_state(state)
        tail = content.splitlines()
        return "\n".join(tail[-max(1, int(lines or 80)) :])
    return ""


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
    desired_spec = selected_model_entry(cfg)["model_spec"]
    running_model = status.get("actual_model_id") or ""
    if not status.get("healthy") or running_model != desired_spec:
        status = start_server(cfg, progress_callback=progress_callback)

    smoke = status.get("smoke_tests") if isinstance(status.get("smoke_tests"), dict) else {}
    if not smoke.get("passed"):
        _emit_progress(progress_callback, "Running llama.cpp smoke tests...")
        smoke = run_smoke_tests(cfg)
        status = get_status(cfg)
        status["smoke_tests"] = smoke

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
    settings = agent_runtime_settings(config)
    return {
        "provider": LLAMA_CPP_PROVIDER,
        "api_mode": "chat_completions",
        "base_url": status["base_url"],
        "api_key": "no-key-required",
        "source": "managed:llama-cpp",
        "model": actual_model,
        "parallel_tool_calls": bool(settings.get("parallel_tool_calls", False)),
        "parser_chain": list(settings.get("parser_chain") or _DEFAULT_PARSER_CHAIN),
    }


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
