from hermes_cli.llama_cpp import binary_candidates, build_server_command, get_engine_config


def test_build_server_command_uses_hf_model_spec(tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    command = build_server_command(
        binary_path=binary_path,
        config={
            "local_engines": {
                "llama_cpp": {
                    "selected_tier": "large",
                    "model_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
                    "quant": "MXFP4_MOE",
                }
            }
        },
    )

    assert command[:3] == [
        str(binary_path),
        "-hf",
        "unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE",
    ]
    assert "--reasoning-budget" in command
    assert command[command.index("--reasoning-budget") + 1] == "0"


def test_build_server_command_accepts_preparsed_engine_config(tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("", encoding="utf-8")

    engine_cfg = get_engine_config(
        {
            "local_engines": {
                "llama_cpp": {
                    "selected_tier": "large",
                    "model_repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
                    "quant": "MXFP4_MOE",
                }
            }
        }
    )

    command = build_server_command(binary_path=binary_path, config=engine_cfg)

    assert command[:3] == [
        str(binary_path),
        "-hf",
        "unsloth/Qwen3.5-35B-A3B-GGUF:MXFP4_MOE",
    ]
    assert command[command.index("--reasoning-budget") + 1] == "0"


def test_binary_candidates_prefers_path_server_before_managed_binary(monkeypatch, tmp_path):
    env_binary = tmp_path / "env-llama-server"
    path_binary = tmp_path / "path-llama-server"
    state_binary = tmp_path / "state-llama-server"
    managed_binary = tmp_path / "managed-llama-server"

    monkeypatch.setenv("HERMES_LLAMA_CPP_BINARY", str(env_binary))
    monkeypatch.setattr("hermes_cli.llama_cpp.load_state", lambda: {"binary_path": str(state_binary)})
    monkeypatch.setattr("hermes_cli.llama_cpp.get_engine_bin_dir", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.llama_cpp.get_engine_binary_name", lambda: managed_binary.name)
    monkeypatch.setattr("hermes_cli.llama_cpp.shutil.which", lambda _name: str(path_binary))

    candidates = binary_candidates()

    assert candidates == [env_binary, path_binary, state_binary, managed_binary]
