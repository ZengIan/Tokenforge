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
TIMEOUT = httpx.Timeout(8.0)
# 浏览器化请求头:ModelScope 的 WAF 常拦截非浏览器 UA / 缺 Referer 的请求 (403)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://modelscope.cn/models",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

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


def _extract_models(data: Any) -> list[dict]:
    """从 ModelScope 多种可能的返回结构中提取模型列表。"""
    if not isinstance(data, dict):
        return []
    d = data.get("Data") or data.get("data") or {}
    if not isinstance(d, dict):
        return []
    # 常见结构: Data.Model.Models / Data.Models / Data.model.models
    for path in (("Model", "Models"), ("model", "models")):
        node = d.get(path[0])
        if isinstance(node, dict):
            models = node.get(path[1])
            if isinstance(models, list):
                return models
    for key in ("Models", "models"):
        if isinstance(d.get(key), list):
            return d[key]
    return []


def _model_to_result(m: dict) -> dict[str, Any]:
    # 兼容字段大小写差异
    path = m.get("Path") or m.get("path") or ""
    name = m.get("Name") or m.get("name") or ""
    model_id = f"{path}/{name}".strip("/")
    tasks = m.get("Tasks") or m.get("tasks") or []
    task = ""
    if tasks and isinstance(tasks, list) and isinstance(tasks[0], dict):
        task = tasks[0].get("Name") or tasks[0].get("name") or ""
    # 官方仓库大小 (StorageSize, 字节) -> 十进制 GB, 与官网显示一致
    storage = m.get("StorageSize") or m.get("storageSize")
    weight_size_gb = round(storage / 1e9, 2) if isinstance(storage, (int, float)) and storage > 0 else None
    return {
        "model_id": model_id,
        "name": name,
        "chinese_name": m.get("ChineseName") or m.get("chineseName") or name,
        "params_b": params_from_name(model_id),
        "precision": precision_from_name(model_id),
        "task": task,
        "downloads": m.get("Downloads") or m.get("downloads") or 0,
        "weight_size_gb": weight_size_gb,
    }


# 官网搜索用的接口在前,SDK 列表接口兜底
_SEARCH_ENDPOINTS = (
    "https://modelscope.cn/api/v1/dolphin/models",
    f"{MS_BASE}/models",
)


async def search_models(query: str, limit: int = 10) -> list[dict[str, Any]]:
    cache_key = f"search:{query}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    body = {
        "Name": query,
        "PageSize": limit,
        "PageNumber": 1,
        "SortBy": "Default",
        "Criterion": [],
        "SingleCriterion": [],
        "Target": "",
    }
    last_err: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
            for url in _SEARCH_ENDPOINTS:
                try:
                    resp = await client.put(url, json=body)
                    resp.raise_for_status()
                    models = _extract_models(resp.json())
                    if models:
                        results = [_model_to_result(m) for m in models]
                        _cache_set(cache_key, results)
                        return results
                except Exception as exc:  # 试下一个端点
                    last_err = str(exc)
                    continue
    except Exception as exc:
        last_err = str(exc)

    # 全部失败/无结果 -> 降级,允许用户手动输入参数
    return [{"error": last_err or "未找到模型", "model_id": query, "params_b": params_from_name(query)}]


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
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
            url = f"{MS_BASE}/models/{model_id}/repo?Revision=master&FilePath=config.json"
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

            # 真实权重大小:汇总仓库内权重文件体积(官方口径,十进制 GB)
            spec.weight_size_gb = await _fetch_weight_size_gb(client, model_id)
    except Exception:
        config_found = False

    out = spec.model_dump()
    out["config_found"] = config_found
    _cache_set(cache_key, out)
    return out


# 计入权重大小的文件后缀
_WEIGHT_EXTS = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


async def _fetch_weight_size_gb(client: httpx.AsyncClient, model_id: str) -> Optional[float]:
    """从 ModelScope 仓库文件列表汇总权重文件体积,返回十进制 GB。"""
    try:
        url = f"{MS_BASE}/models/{model_id}/repo/files?Revision=master&Recursive=true"
        resp = await client.get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
        files = (data.get("Data") or {}).get("Files") or []
        total = 0
        for f in files:
            name = (f.get("Path") or f.get("Name") or "").lower()
            if name.endswith(_WEIGHT_EXTS):
                total += int(f.get("Size", 0) or 0)
        if total <= 0:
            return None
        return round(total / 1e9, 2)  # 与 ModelScope 页面一致的十进制 GB
    except Exception:
        return None
