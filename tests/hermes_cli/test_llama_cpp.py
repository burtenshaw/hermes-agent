import os
from unittest.mock import patch

import yaml

from hermes_cli import runtime_provider as rp
from hermes_cli.config import load_config
from hermes_cli.llama_cpp import binary_candidates, build_server_command
from hermes_cli.models import validate_requested_model


def test_load_config_canonicalizes_llama_cpp_aliases(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        for provider in ("llama.cpp", "local"):
            (tmp_path / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "provider": provider,
                            "default": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_config()

            assert config["model"]["provider"] == "llama-cpp"



def test_build_server_command_uses_hf_model_spec(tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    command = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "selected_tier": "large",
                    "model_repo": "ggml-org/gemma-4-26B-A4B-it-GGUF",
                    "quant": "Q4_K_M",
                }
            }
        },
    )

    assert command[:3] == [
        str(binary_path),
        "-hf",
        "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
    ]
    assert command[command.index("--reasoning-format") + 1] == "none"
    assert command[command.index("--reasoning-budget") + 1] == "0"


def test_build_server_command_uses_deepseek_reasoning_format_when_enabled(tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    command = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "selected_tier": "balanced",
                    "model_repo": "ggml-org/gemma-4-E4B-it-GGUF",
                    "quant": "Q4_K_M",
                    "reasoning_budget": -1,
                }
            }
        },
    )

    assert command[command.index("--reasoning-format") + 1] == "deepseek"


def test_binary_candidates_prefers_path_server_before_managed_binary(monkeypatch, tmp_path):
    env_binary = tmp_path / "env-llama-server"
    path_binary = tmp_path / "path-llama-server"
    state_binary = tmp_path / "state-llama-server"
    managed_binary = tmp_path / "managed-llama-server"

    monkeypatch.setenv("HERMES_LLAMA_CPP_BINARY", str(env_binary))
    monkeypatch.setattr(
        "hermes_cli.llama_cpp.load_state",
        lambda: {"binary_path": str(state_binary)},
    )
    monkeypatch.setattr("hermes_cli.llama_cpp.get_engine_bin_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.llama_cpp.get_engine_binary_name",
        lambda: managed_binary.name,
    )
    monkeypatch.setattr(
        "hermes_cli.llama_cpp.shutil.which",
        lambda _name: str(path_binary),
    )

    candidates = binary_candidates()

    assert candidates == [env_binary, path_binary, state_binary, managed_binary]


def test_validate_requested_model_enforces_curated_llama_cpp_allowlist():
    accepted = validate_requested_model(
        "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        "llama-cpp",
    )
    rejected = validate_requested_model(
        "some-random/Model-GGUF:Q4_K_M",
        "llama-cpp",
    )

    assert accepted["accepted"] is True
    assert accepted["persist"] is True
    assert rejected["accepted"] is False
    assert rejected["persist"] is False
    assert "curated llama.cpp allowlist" in rejected["message"]


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

    for requested, configured_provider in (
        ("llama-cpp", "llama-cpp"),
        ("local", "local"),
    ):
        monkeypatch.setattr(
            rp,
            "resolve_requested_provider",
            lambda requested=None, _requested=requested: _requested,
        )
        monkeypatch.setattr(
            rp,
            "load_config",
            lambda _provider=configured_provider: {
                "model": {
                    "provider": _provider,
                    "default": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
                }
            },
        )

        resolved = rp.resolve_runtime_provider(requested=requested)

        assert resolved["provider"] == "llama-cpp"
        assert resolved["api_mode"] == "chat_completions"
        assert resolved["base_url"] == "http://127.0.0.1:8081/v1"
        assert resolved["api_key"] == "no-key-required"
        assert resolved["requested_provider"] == requested


def test_custom_local_provider_requires_explicit_custom_prefix(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "Local",
                    "base_url": "http://1.2.3.4:1234/v1",
                    "api_key": "local-provider-key",
                }
            ]
        },
    )
    monkeypatch.setattr(
        rp.auth_mod,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for explicit named custom providers"
            )
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="custom:local")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "http://1.2.3.4:1234/v1"
    assert resolved["api_key"] == "local-provider-key"
    assert resolved["requested_provider"] == "custom:local"
