"""Model/config helpers for managed llama.cpp support."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hermes_cli.llama_cpp_common import (
    LLAMA_CPP_DEFAULT_PORT,
    LLAMA_CPP_PROVIDER,
    LLAMA_CPP_PROVIDER_ALIASES,
    LlamaCppError,
    _merge_dict,
)

_DEFAULT_CONTEXT_LENGTH = 32768
_DEFAULT_PARSER_CHAIN = ["llama3_json", "hermes"]
_DEFAULT_GPU_LAYERS = -1
_DEFAULT_STARTUP_STALL_TIMEOUT_SECONDS = 600
_VALID_ACCELERATIONS = {
    "auto",
    "cpu",
    "metal",
    "cuda",
}
_GPU_ACCELERATIONS = _VALID_ACCELERATIONS - {"auto", "cpu"}

CURATED_MODELS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "tier": "tiny",
        "model_repo": "ggml-org/gemma-4-E2B-it-GGUF",
        "quant": "Q8_0",
        "context_length": 131072,
        "template_strategy": "native",
        "parser_chain": list(_DEFAULT_PARSER_CHAIN),
    },
    "balanced": {
        "tier": "balanced",
        "model_repo": "ggml-org/gemma-4-E4B-it-GGUF",
        "quant": "Q4_K_M",
        "context_length": 131072,
        "template_strategy": "native",
        "parser_chain": list(_DEFAULT_PARSER_CHAIN),
    },
    "large": {
        "tier": "large",
        "model_repo": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "quant": "Q4_K_M",
        "context_length": 262144,
        "template_strategy": "native",
        "parser_chain": list(_DEFAULT_PARSER_CHAIN),
    },
}

_CURATED_SPECS = {
    f"{entry['model_repo']}:{entry['quant']}": dict(entry)
    for entry in CURATED_MODELS.values()
}


def default_engine_config() -> Dict[str, Any]:
    return {
        "managed": True,
        "auto_start": True,
        "port": LLAMA_CPP_DEFAULT_PORT,
        "model": "",
        "acceleration": "auto",
        "gpu_layers": _DEFAULT_GPU_LAYERS,
        "context_length": 0,
        "startup_stall_timeout_seconds": _DEFAULT_STARTUP_STALL_TIMEOUT_SECONDS,
        "reasoning_budget": 0,
        "reasoning_format": "deepseek",
        "template_strategy": "native",
        "template_file": "",
        "parallel_tool_calls": True,
    }


def is_llama_cpp_provider(value: Optional[str]) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in LLAMA_CPP_PROVIDER_ALIASES


def normalize_acceleration(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _VALID_ACCELERATIONS:
        return normalized
    return "auto"


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
        elif any(
            key in config
            for key in (
                *default_engine_config().keys(),
                "selected_tier",
                "model_repo",
                "quant",
            )
        ):
            engine_cfg = config
    merged = _merge_dict(default_engine_config(), engine_cfg)
    try:
        merged["port"] = int(merged.get("port") or LLAMA_CPP_DEFAULT_PORT)
    except Exception:
        merged["port"] = LLAMA_CPP_DEFAULT_PORT
    raw_context_length = merged.get("context_length")
    try:
        if raw_context_length in (None, ""):
            merged["context_length"] = 0
        else:
            merged["context_length"] = max(0, int(raw_context_length))
    except Exception:
        merged["context_length"] = 0
    try:
        merged["reasoning_budget"] = int(merged.get("reasoning_budget"))
    except Exception:
        merged["reasoning_budget"] = 0
    merged["reasoning_format"] = str(merged.get("reasoning_format") or "deepseek").strip().lower() or "deepseek"
    merged["selected_tier"] = str(merged.get("selected_tier") or "").strip().lower()
    merged["template_strategy"] = str(merged.get("template_strategy") or "native").strip().lower()
    merged["template_file"] = str(merged.get("template_file") or "").strip()
    merged["model_repo"] = str(merged.get("model_repo") or "").strip()
    merged["quant"] = str(merged.get("quant") or "").strip()
    merged["model"] = canonical_model_value(merged.get("model"))
    if not merged["model"]:
        if merged["selected_tier"] in CURATED_MODELS:
            merged["model"] = merged["selected_tier"]
        else:
            merged["model"] = spec_string(merged["model_repo"], merged["quant"])
    resolved_model = _resolve_model_value(merged["model"])
    merged["model"] = resolved_model["model"]
    merged["selected_tier"] = resolved_model["selected_tier"]
    merged["model_repo"] = resolved_model["model_repo"]
    merged["quant"] = resolved_model["quant"]
    merged["managed"] = bool(merged.get("managed", True))
    merged["auto_start"] = bool(merged.get("auto_start", True))
    merged["acceleration"] = normalize_acceleration(merged.get("acceleration"))
    try:
        merged["gpu_layers"] = int(merged.get("gpu_layers", _DEFAULT_GPU_LAYERS))
    except Exception:
        merged["gpu_layers"] = _DEFAULT_GPU_LAYERS
    if merged["gpu_layers"] < -1:
        merged["gpu_layers"] = _DEFAULT_GPU_LAYERS
    try:
        merged["startup_stall_timeout_seconds"] = int(
            merged.get(
                "startup_stall_timeout_seconds",
                _DEFAULT_STARTUP_STALL_TIMEOUT_SECONDS,
            )
        )
    except Exception:
        merged["startup_stall_timeout_seconds"] = _DEFAULT_STARTUP_STALL_TIMEOUT_SECONDS
    merged["startup_stall_timeout_seconds"] = max(60, merged["startup_stall_timeout_seconds"])
    merged["parallel_tool_calls"] = bool(merged.get("parallel_tool_calls", False))
    merged.pop("streaming_tool_calls", None)
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


def canonical_model_value(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.lower()
    if normalized in CURATED_MODELS:
        return normalized
    parsed = parse_model_spec(text)
    return spec_string(parsed.get("model_repo", ""), parsed.get("quant", ""))


def _resolve_model_value(value: Optional[str]) -> Dict[str, str]:
    canonical = canonical_model_value(value)
    resolved = {
        "model": canonical,
        "selected_tier": "",
        "model_repo": "",
        "quant": "",
    }
    if not canonical:
        return resolved
    if canonical in CURATED_MODELS:
        entry = CURATED_MODELS[canonical]
        resolved["selected_tier"] = canonical
        resolved["model_repo"] = str(entry.get("model_repo") or "").strip()
        resolved["quant"] = str(entry.get("quant") or "").strip()
        return resolved

    parsed = parse_model_spec(canonical)
    resolved["model_repo"] = parsed.get("model_repo", "")
    resolved["quant"] = parsed.get("quant", "")
    curated = _CURATED_SPECS.get(canonical)
    if curated:
        resolved["selected_tier"] = str(curated.get("tier") or "").strip().lower()
    return resolved


def curated_model_specs() -> list[str]:
    return list(_CURATED_SPECS.keys())


def curated_entry_for_tier(tier: str) -> Dict[str, Any]:
    normalized = str(tier or "").strip().lower()
    if normalized not in CURATED_MODELS:
        raise LlamaCppError(f"Unknown llama.cpp tier '{tier}'.")
    return dict(CURATED_MODELS[normalized])


def curated_entry_for_spec(model_repo: str, quant: str) -> Optional[Dict[str, Any]]:
    return _CURATED_SPECS.get(spec_string(model_repo, quant))


def recommended_parser_chain(config: Optional[Dict[str, Any]] = None) -> list[str]:
    return list(selected_model_entry(config).get("parser_chain") or _DEFAULT_PARSER_CHAIN)


def effective_reasoning_budget(config: Optional[Dict[str, Any]] = None) -> int:
    cfg = get_engine_config(config)
    try:
        budget = int(cfg.get("reasoning_budget", 0))
        return max(0, budget)
    except Exception:
        return 0


def selected_model_entry(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    model_repo = cfg.get("model_repo", "").strip()
    quant = cfg.get("quant", "").strip()
    if model_repo:
        current = curated_entry_for_spec(model_repo, quant)
        if current is None:
            return {
                "tier": cfg.get("selected_tier") or "",
                "model_repo": model_repo,
                "quant": quant,
                "model_spec": spec_string(model_repo, quant),
                "context_length": cfg.get("context_length") or _DEFAULT_CONTEXT_LENGTH,
                "template_strategy": cfg.get("template_strategy", "native"),
                "parser_chain": list(_DEFAULT_PARSER_CHAIN),
            }
        selected = dict(current)
    else:
        tier = cfg.get("selected_tier") or "balanced"
        if tier not in CURATED_MODELS:
            tier = "balanced"
        selected = curated_entry_for_tier(tier)

    selected["context_length"] = cfg.get("context_length") or selected.get("context_length") or _DEFAULT_CONTEXT_LENGTH
    selected["model_spec"] = spec_string(selected.get("model_repo", ""), selected.get("quant", ""))
    return selected


def agent_runtime_settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    return {
        "parallel_tool_calls": bool(cfg.get("parallel_tool_calls", False)),
        "parser_chain": recommended_parser_chain(cfg),
    }


def effective_gpu_layers(
    config: Optional[Dict[str, Any]] = None,
    *,
    resolved_acceleration: Optional[str] = None,
) -> int:
    cfg = get_engine_config(config)
    acceleration = normalize_acceleration(
        resolved_acceleration if resolved_acceleration is not None else cfg.get("acceleration")
    )
    if acceleration not in _GPU_ACCELERATIONS:
        return 0
    try:
        raw = int(cfg.get("gpu_layers", _DEFAULT_GPU_LAYERS))
    except Exception:
        raw = _DEFAULT_GPU_LAYERS
    if raw == 0:
        return 0
    if raw < 0:
        return 999
    return raw


def ensure_engine_config_section(config: Dict[str, Any]) -> Dict[str, Any]:
    local_engines = config.setdefault("local_engines", {})
    if not isinstance(local_engines, dict):
        local_engines = {}
        config["local_engines"] = local_engines
    engine_cfg = local_engines.setdefault("llama_cpp", {})
    if not isinstance(engine_cfg, dict):
        engine_cfg = {}
        local_engines["llama_cpp"] = engine_cfg
    return engine_cfg


def sync_config_model_fields(config: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = get_engine_config(config)
    if runtime is None:
        from hermes_cli.llama_cpp import get_status

        runtime = get_status(cfg)
    entry = selected_model_entry(cfg)
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {"default": model_cfg} if model_cfg else {}
    model_cfg["provider"] = LLAMA_CPP_PROVIDER
    model_cfg["base_url"] = runtime_base_url(cfg)
    model_cfg["api_mode"] = "chat_completions"
    model_cfg["default"] = str(runtime.get("model_spec") or entry["model_spec"])
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
    engine_cfg = ensure_engine_config_section(config)
    curated = curated_entry_for_tier(tier)
    engine_cfg["managed"] = True
    engine_cfg["auto_start"] = True
    selected_model = (
        spec_string(model_repo or curated["model_repo"], quant or curated["quant"])
        if (model_repo or quant)
        else tier
    )
    engine_cfg["model"] = canonical_model_value(selected_model)
    engine_cfg["context_length"] = int(context_length or curated.get("context_length") or _DEFAULT_CONTEXT_LENGTH)
    if "reasoning_budget" in engine_cfg:
        try:
            engine_cfg["reasoning_budget"] = int(engine_cfg.get("reasoning_budget"))
        except Exception:
            engine_cfg["reasoning_budget"] = 0
    else:
        engine_cfg["reasoning_budget"] = 0
    engine_cfg["template_strategy"] = curated.get("template_strategy", "native")
    for legacy_key in ("selected_tier", "model_repo", "quant"):
        engine_cfg.pop(legacy_key, None)
    return config
