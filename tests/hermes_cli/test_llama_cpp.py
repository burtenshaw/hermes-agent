import os
from unittest.mock import patch

import yaml

from hermes_cli import runtime_provider as rp
from hermes_cli.config import load_config
from hermes_cli.llama_cpp import build_server_command, get_engine_config


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
