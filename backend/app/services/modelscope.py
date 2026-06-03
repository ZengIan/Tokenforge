"""ModelScope API proxy: model search + architecture metadata extraction.

The backend only proxies/normalises public ModelScope endpoints (PRD §9.1) and
caches results in-memory. On any network failure we degrade gracefully so the
frontend can fall back to manual parameter entry (PRD §6.2).
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

import httpx

from ..models.schemas import ModelSpec

MS_BASE = "https://modelscope.cn/api/v1"
TIMEOUT = httpx.Timeout(5.0)
_HEADERS = {"User-Agent": "tokenforge/1.0"}

# very small TTL cache: key -> (expires_at, value)
_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 300.0


def _cache_get(key: str) -> Optional[Any]:
    hit = _CACHE.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.time() + _TTL, value)


_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def params_from_name(model_id: str) -> Optional[float]:
    """Best-effort '...-7B-...' -> 7.0 billion parameters."""
    m = _PARAM_RE.search(model_id.replace("_", "-"))
    if m:
        return float(m.group(1))
    return None


_PRECISION_HINTS = [
    ("int4", "INT4"),
    ("gptq", "GPTQ"),
    ("awq", "AWQ"),
    ("int8", "INT8"),
    ("fp8", "FP8"),
    ("bf16", "BF16"),
    ("fp16", "FP16"),
]


def precision_from_name(model_id: str) -> str:
    low = model_id.lower()
    for needle, label in _PRECISION_HINTS:
        if needle in low:
            return label
    return "FP16"


async def search_models(query: str, limit: int = 10) -> list[dict[str, Any]]:
    cache_key = f"search:{query}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    results: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
            resp = await client.put(
                f"{MS_BASE}/models",
                json={"Name": query, "PageSize": limit, "PageNumber": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            models = (data.get("Data") or {}).get("Model", {}).get("Models") or []
            for m in models:
                model_id = f"{m.get('Path', '')}/{m.get('Name', '')}".strip("/")
                results.append(
                    {
                        "model_id": model_id,
                        "name": m.get("Name", ""),
                        "chinese_name": m.get("ChineseName", "") or m.get("Name", ""),
                        "params_b": params_from_name(model_id),
                        "precision": precision_from_name(model_id),
                        "task": m.get("Tasks", [{}])[0].get("Name", "")
                        if m.get("Tasks")
                        else "",
                        "downloads": m.get("Downloads", 0),
                    }
                )
    except Exception as exc:  # network/parse failure -> graceful degrade
        return [{"error": str(exc), "model_id": query, "params_b": params_from_name(query)}]

    _cache_set(cache_key, results)
    return results


async def get_model_config(model_id: str) -> dict[str, Any]:
    """Fetch config.json from a model repo and map to ModelSpec fields."""
    cache_key = f"config:{model_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    spec = ModelSpec(model_id=model_id, params_b=params_from_name(model_id) or 7.0)
    spec.precision = precision_from_name(model_id)
    config_found = False

    try:
        url = f"{MS_BASE}/models/{model_id}/repo?Revision=master&FilePath=config.json"
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                cfg = resp.json()
                if isinstance(cfg, dict) and cfg:
                    config_found = True
                    spec.hidden_size = int(cfg.get("hidden_size", spec.hidden_size))
                    spec.num_layers = int(cfg.get("num_hidden_layers", spec.num_layers))
                    spec.num_attention_heads = int(
                        cfg.get("num_attention_heads", spec.num_attention_heads)
                    )
                    if cfg.get("num_key_value_heads") is not None:
                        spec.num_key_value_heads = int(cfg["num_key_value_heads"])
                    spec.vocab_size = int(cfg.get("vocab_size", spec.vocab_size))
    except Exception:
        config_found = False

    out = spec.model_dump()
    out["config_found"] = config_found
    _cache_set(cache_key, out)
    return out
