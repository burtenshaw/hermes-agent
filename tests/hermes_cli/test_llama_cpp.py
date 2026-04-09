import os
from unittest.mock import patch

import pytest
import yaml

import hermes_cli.llama_cpp as llama_cpp
from hermes_cli import runtime_provider as rp
from hermes_cli.config import load_config
from hermes_cli.llama_cpp import (
    agent_runtime_settings,
    build_server_command,
    ensure_engine_config_section,
    get_engine_config,
)
from hermes_cli.llama_cpp_common import load_state, save_state


def test_load_config_normalizes_llama_cpp_provider_and_model(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "provider": "local",
                        "default": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
                    },
                    "local_engines": {
                        "llama_cpp": {
                            "selected_tier": "tiny",
                            "model_repo": "ggml-org/gemma-4-E2B-it-GGUF",
                            "quant": "Q8_0",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        config = load_config()
        engine_cfg = get_engine_config(config)

        assert config["model"]["provider"] == "llama-cpp"
        assert config["local_engines"]["llama_cpp"]["model"] == "tiny"
        assert "selected_tier" not in config["local_engines"]["llama_cpp"]
        assert "model_repo" not in config["local_engines"]["llama_cpp"]
        assert "quant" not in config["local_engines"]["llama_cpp"]
        assert engine_cfg["selected_tier"] == "tiny"
        assert engine_cfg["model_repo"] == "ggml-org/gemma-4-E2B-it-GGUF"
        assert engine_cfg["quant"] == "Q8_0"


def test_build_server_command_uses_model_key_and_reasoning_format(tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    command = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "model": "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
                }
            }
        },
    )
    override = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "model": "balanced",
                    "reasoning_format": "none",
                }
            }
        },
    )

    assert command[:3] == [
        str(binary_path),
        "-hf",
        "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
    ]
    assert command[command.index("--reasoning-format") + 1] == "deepseek"
    assert override[override.index("--reasoning-format") + 1] == "none"


def test_get_engine_config_normalizes_acceleration_fields():
    engine_cfg = get_engine_config(
        {
            "local_engines": {
                "llama_cpp": {
                    "acceleration": "CUDA",
                    "gpu_layers": "12",
                    "startup_stall_timeout_seconds": "30",
                }
            }
        }
    )

    assert engine_cfg["acceleration"] == "cuda"
    assert engine_cfg["gpu_layers"] == 12
    assert engine_cfg["startup_stall_timeout_seconds"] == 60


def test_get_engine_config_unsupported_acceleration_falls_back_to_auto():
    engine_cfg = get_engine_config(
        {
            "local_engines": {
                "llama_cpp": {
                    "acceleration": "rocm",
                }
            }
        }
    )

    assert engine_cfg["acceleration"] == "auto"


def test_build_server_command_adds_gpu_layers_for_gpu_acceleration(tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    gpu_command = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "model": "balanced",
                    "acceleration": "cuda",
                    "gpu_layers": -1,
                }
            }
        },
        resolved_acceleration="cuda",
    )
    cpu_command = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "model": "balanced",
                    "acceleration": "cpu",
                }
            }
        },
        resolved_acceleration="cpu",
    )

    assert gpu_command[gpu_command.index("--n-gpu-layers") + 1] == "all"
    assert "--n-gpu-layers" not in cpu_command


def test_agent_runtime_settings_collects_llama_runtime_flags():
    settings = agent_runtime_settings(
        {
            "local_engines": {
                "llama_cpp": {
                    "model": "balanced",
                    "parallel_tool_calls": True,
                }
            }
        }
    )

    assert settings["parallel_tool_calls"] is True
    assert settings["parser_chain"] == ["llama3_json", "hermes"]


def test_ensure_engine_config_section_repairs_malformed_config():
    config = {"local_engines": []}

    section = ensure_engine_config_section(config)
    section["model"] = "tiny"

    assert isinstance(config["local_engines"], dict)
    assert config["local_engines"]["llama_cpp"]["model"] == "tiny"


def test_resolve_release_asset_prefers_cuda(monkeypatch):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Windows")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        llama_cpp,
        "_fetch_release_metadata",
        lambda _version: {
            "assets": [
                {
                    "name": "llama-b8720-bin-win-cpu-x64.zip",
                    "browser_download_url": "https://example.invalid/cpu.zip",
                },
                {
                    "name": "llama-b8720-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.invalid/cuda.zip",
                },
            ]
        },
    )

    asset, checksum = llama_cpp._resolve_release_asset("latest", acceleration="cuda")

    assert asset["name"] == "llama-b8720-bin-win-cuda-12.4-x64.zip"
    assert checksum is None


def test_resolve_acceleration_prefers_metal_on_macos_arm64_auto(monkeypatch):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "arm64")

    resolved = llama_cpp.resolve_acceleration(
        {
            "local_engines": {
                "llama_cpp": {
                    "acceleration": "auto",
                }
            }
        }
    )

    assert resolved == "metal"


def test_resolve_release_asset_prefers_arm64_archive_for_metal(monkeypatch):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        llama_cpp,
        "_fetch_release_metadata",
        lambda _version: {
            "assets": [
                {
                    "name": "llama-b8720-bin-macos-x64.tar.gz",
                    "browser_download_url": "https://example.invalid/x64.tar.gz",
                },
                {
                    "name": "llama-b8720-bin-macos-arm64.tar.gz",
                    "browser_download_url": "https://example.invalid/arm64.tar.gz",
                },
            ]
        },
    )

    asset, checksum = llama_cpp._resolve_release_asset("latest", acceleration="metal")

    assert asset["name"] == "llama-b8720-bin-macos-arm64.tar.gz"
    assert checksum is None


def test_resolve_binary_path_prefers_managed_install_over_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    managed_binary = tmp_path / "local_engines" / "llama_cpp" / "bin" / llama_cpp.get_engine_binary_name()
    managed_binary.parent.mkdir(parents=True, exist_ok=True)
    managed_binary.write_text("", encoding="utf-8")

    external_binary = tmp_path / "external-llama-server"
    external_binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(llama_cpp, "_path_binary_path", lambda: external_binary)

    resolved = llama_cpp.resolve_binary_path({"local_engines": {"llama_cpp": {"managed": True}}})

    assert resolved == managed_binary


def test_ensure_binary_installed_reinstalls_managed_binary_when_auto_backend_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    managed_binary = tmp_path / "local_engines" / "llama_cpp" / "bin" / llama_cpp.get_engine_binary_name()
    managed_binary.parent.mkdir(parents=True, exist_ok=True)
    managed_binary.write_text("", encoding="utf-8")
    save_state(
        {
            "binary_path": str(managed_binary),
            "binary_asset_name": "llama-b8720-bin-win-cuda-12.4-x64.zip",
            "requested_acceleration": "auto",
            "binary_acceleration": "cuda",
        }
    )

    installs = []
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(llama_cpp, "_cuda_runtime_available", lambda: False)
    monkeypatch.setattr(
        llama_cpp,
        "_install_release_binary",
        lambda _version, *, acceleration: installs.append(acceleration)
        or (managed_binary, "llama-b8720-bin-linux-x64.zip"),
    )
    monkeypatch.setattr(llama_cpp, "read_binary_version", lambda _path: "b8720")

    binary_path, resolved = llama_cpp.ensure_binary_installed(
        {
            "local_engines": {
                "llama_cpp": {
                    "managed": True,
                    "acceleration": "auto",
                }
            }
        }
    )
    state = load_state()

    assert binary_path == managed_binary
    assert resolved == "cpu"
    assert installs == ["cpu"]
    assert state["binary_asset_name"] == "llama-b8720-bin-linux-x64.zip"
    assert state["binary_source"] == "managed"
    assert "requested_acceleration" not in state
    assert "binary_acceleration" not in state


def test_stop_server_clears_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(llama_cpp, "_pid_is_alive", lambda pid: int(pid) == 123 if pid else False)
    monkeypatch.setattr(llama_cpp, "_utc_now_iso", lambda: "2026-04-09T10:00:00Z")
    save_state(
        {
            "pid": 123,
            "actual_model_id": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
            "props": {"ok": True},
        }
    )

    terminated = []
    monkeypatch.setattr(
        llama_cpp,
        "_terminate_existing_process",
        lambda state, force=False: terminated.append((state.get("pid"), force)) or True,
    )
    monkeypatch.setattr(llama_cpp, "_port_is_listening", lambda _port: True)
    monkeypatch.setattr(llama_cpp, "_kill_server_on_port", lambda _port, force=False: [456])

    result = llama_cpp.stop_server({"local_engines": {"llama_cpp": {"port": 8081}}}, force=True)
    state = load_state()

    assert result["stopped"] is True
    assert result["terminated_pid"] == 123
    assert result["killed_pids"] == [456]
    assert terminated == [(123, True)]
    assert state["pid"] is None
    assert state["actual_model_id"] == ""
    assert state["props"] is None
    assert state["stopped_at"] == "2026-04-09T10:00:00Z"


def test_start_server_uses_stall_timeout_not_fixed_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    class FakeProcess:
        pid = 321

        def poll(self):
            return None

    progress_steps = [
        {
            "phase": "downloading",
            "artifact": "model",
            "current_bytes": 10,
            "total_bytes": 100,
            "message": "Downloading model 10 / 100",
        },
        {
            "phase": "downloading",
            "artifact": "model",
            "current_bytes": 20,
            "total_bytes": 100,
            "message": "Downloading model 20 / 100",
        },
        {
            "phase": "loading",
            "message": "Loading model into memory...",
        },
    ]

    clock = {"now": 0}

    def fake_time():
        return clock["now"]

    def fake_sleep(_seconds):
        clock["now"] += 30

    def fake_progress(_config):
        if progress_steps:
            return progress_steps.pop(0)
        return {
            "phase": "loading",
            "message": "Loading model into memory...",
        }

    monkeypatch.setattr(
        llama_cpp,
        "_probe_server",
        lambda _url: {"healthy": False, "actual_model_id": "", "props": None, "models": []},
    )
    monkeypatch.setattr(llama_cpp, "ensure_binary_installed", lambda _cfg: (binary_path, "cpu"))
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(llama_cpp, "describe_startup_progress", fake_progress)
    monkeypatch.setattr(llama_cpp.time, "time", fake_time)
    monkeypatch.setattr(llama_cpp.time, "sleep", fake_sleep)
    monkeypatch.setattr(llama_cpp, "read_binary_version", lambda _path: "")

    with pytest.raises(llama_cpp.LlamaCppError, match="No startup progress from llama.cpp for 60 seconds"):
        llama_cpp.start_server(
            {
                "local_engines": {
                    "llama_cpp": {
                        "model": "balanced",
                        "startup_stall_timeout_seconds": 60,
                    }
                }
            }
        )


def test_resolve_runtime_provider_llama_cpp_aliases_use_managed_runtime_payload(monkeypatch):
    monkeypatch.setattr(
        rp,
        "llama_cpp_runtime_payload",
        lambda _config, progress_callback=None: {
            "provider": "llama-cpp",
            "api_mode": "chat_completions",
            "base_url": "http://127.0.0.1:8081/v1",
            "api_key": "no-key-required",
            "source": "managed:llama-cpp",
            "model": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
        },
    )

    for requested in ("llama-cpp", "local"):
        monkeypatch.setattr(
            rp,
            "resolve_requested_provider",
            lambda requested=None, _requested=requested: _requested,
        )
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda: {
                "model": {
                    "provider": "llama-cpp",
                    "default": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
                }
            },
        )

        resolved = rp.resolve_runtime_provider(requested=requested)

        assert resolved["provider"] == "llama-cpp"
        assert resolved["requested_provider"] == requested
