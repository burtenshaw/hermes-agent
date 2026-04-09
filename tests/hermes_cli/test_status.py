from types import SimpleNamespace

import hermes_cli.status as status_module
from hermes_cli.status import show_status


def test_show_status_includes_tavily_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-1234567890abcdef")

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert "tvly...cdef" in output


def test_show_status_includes_llama_cpp_acceleration(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_module, "load_config", lambda: {})
    monkeypatch.setattr(
        status_module,
        "get_llama_cpp_status",
        lambda config, check_health=False: {
            "installed": True,
            "healthy": True,
            "smoke_tests": {"passed": True},
            "base_url": "http://127.0.0.1:8081/v1",
            "installed_version": "b8720",
            "binary_path": "/tmp/llama-server",
            "model_spec": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
            "actual_model_id": "ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M",
            "requested_acceleration": "auto",
            "resolved_acceleration": "metal",
            "configured_gpu_layers": -1,
            "gpu_layers": 999,
            "reasoning_budget": 0,
            "template_strategy": "native",
            "parallel_tool_calls": True,
            "started_at": "2026-04-09T10:00:00Z",
            "process_running": True,
        },
    )

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Acceleration" in output
    assert "auto -> metal (all gpu layers)" in output
