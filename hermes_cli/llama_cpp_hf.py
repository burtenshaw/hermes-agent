"""Hugging Face discovery helpers for managed llama.cpp setup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

from hermes_cli.llama_cpp_common import LlamaCppError

_HF_MODELS_API_URL = "https://huggingface.co/api/models"
_HF_SEARCH_DEFAULT_LIMIT = 12
_HF_SEARCH_DEFAULT_PIPELINE_TAG = "image-text-to-text"
_HF_SEARCH_DEFAULT_NUM_PARAMETERS = "min:0,max:32B"
_GGUF_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
_PREFERRED_QUANTS = (
    "Q4_K_M",
    "Q6_K",
    "Q8_0",
    "F16",
    "BF16",
)


def _hf_api_get(path: str = "", *, params: Optional[Dict[str, Any]] = None) -> Any:
    url = _HF_MODELS_API_URL if not path else f"{_HF_MODELS_API_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": "hermes-agent/llama-cpp"},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        raise LlamaCppError(f"Hugging Face request failed ({exc.response.status_code}).") from exc
    except requests.RequestException as exc:
        raise LlamaCppError(f"Could not reach Hugging Face Hub: {exc}") from exc


def search_huggingface_models(
    query: str = "",
    *,
    limit: int = _HF_SEARCH_DEFAULT_LIMIT,
    pipeline_tag: str = _HF_SEARCH_DEFAULT_PIPELINE_TAG,
    num_parameters: str = _HF_SEARCH_DEFAULT_NUM_PARAMETERS,
    sort: str = "trendingScore",
) -> list[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "apps": "llama.cpp",
        "sort": sort,
        "limit": max(1, min(int(limit or _HF_SEARCH_DEFAULT_LIMIT), 50)),
    }
    query = str(query or "").strip()
    if query:
        params["search"] = query
    pipeline_tag = str(pipeline_tag or "").strip()
    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag
    num_parameters = str(num_parameters or "").strip()
    if num_parameters:
        params["num_parameters"] = num_parameters

    payload = _hf_api_get(params=params)
    if not isinstance(payload, list):
        raise LlamaCppError("Hugging Face model search returned malformed data.")

    results: list[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = str(item.get("id") or item.get("modelId") or "").strip()
        if not repo_id:
            continue
        results.append(
            {
                "id": repo_id,
                "pipeline_tag": str(item.get("pipeline_tag") or "").strip(),
                "downloads": int(item.get("downloads") or 0),
                "likes": int(item.get("likes") or 0),
                "author": str(item.get("author") or "").strip(),
                "last_modified": str(item.get("lastModified") or "").strip(),
            }
        )
    return results


def _common_prefix_tokens(values: list[str]) -> list[str]:
    tokenized = [value.split("-") for value in values if value]
    if not tokenized:
        return []
    prefix: list[str] = []
    for parts in zip(*tokenized):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return prefix


def _extract_repo_quants(filenames: list[str]) -> list[str]:
    candidates = [
        Path(name).name
        for name in filenames
        if isinstance(name, str)
        and name.lower().endswith(".gguf")
        and "/" not in name
        and "mmproj" not in name.lower()
        and not _GGUF_SHARD_RE.search(name)
    ]
    if not candidates:
        return []

    stems = [Path(name).stem for name in candidates]
    prefix_tokens = _common_prefix_tokens(stems)
    prefix_len = len(prefix_tokens)
    quants: list[str] = []
    for stem in stems:
        tokens = stem.split("-")
        suffix_tokens = tokens[prefix_len:] if prefix_len < len(tokens) else []
        quant = "-".join(suffix_tokens).strip("-_.")
        if not quant:
            quant = stem
        if quant not in quants:
            quants.append(quant)
    return quants


def preferred_quant(quants: Iterable[str]) -> str:
    values = [str(quant or "").strip() for quant in quants if str(quant or "").strip()]
    if not values:
        return ""
    lowered = {value.lower(): value for value in values}
    for preferred in _PREFERRED_QUANTS:
        match = lowered.get(preferred.lower())
        if match:
            return match
    return values[0]


def list_huggingface_gguf_quants(repo_id: str) -> list[str]:
    repo = str(repo_id or "").strip()
    if not repo:
        return []
    payload = _hf_api_get(repo, params={"expand": "siblings"})
    siblings = payload.get("siblings") if isinstance(payload, dict) else None
    if not isinstance(siblings, list):
        return []
    filenames = []
    for item in siblings:
        if isinstance(item, dict) and item.get("rfilename"):
            filenames.append(str(item["rfilename"]))
    quants = _extract_repo_quants(filenames)
    preferred = preferred_quant(quants)
    if preferred:
        return [preferred] + [quant for quant in quants if quant != preferred]
    return quants
