"""ModelScope API proxy: model search + architecture metadata extraction.

The backend only proxies/normalises public ModelScope endpoints (PRD §9.1) and
caches results in-memory. On any network failure we degrade gracefully so the
frontend can fall back to manual parameter entry (PRD §6.2).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import httpx

from ..models.schemas import ModelSpec

MS_BASE = "https://modelscope.cn/api/v1"
TIMEOUT = httpx.Timeout(20.0)
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


def params_from_storage(storage_bytes: Optional[float], model_id: str) -> Optional[float]:
    """正则无法提取参数量时，从 StorageSize 推算（GB ≈ B for FP8, GB/2 for FP16/BF16）。"""
    if not isinstance(storage_bytes, (int, float)) or storage_bytes <= 0:
        return None
    gb = storage_bytes / 1e9
    low = model_id.lower()
    if any(k in low for k in ("fp8", "int8", "w8a8", "quant", "gptq", "awq")):
        bytes_per_param = 1
    elif any(k in low for k in ("fp16", "bf16", "float16", "bfloat16")):
        bytes_per_param = 2
    else:
        bytes_per_param = 2  # default assumption
    return round(gb / bytes_per_param, 2)


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
    # 官方仓库大小 (StorageSize, 字节) -> 十进制 GB, 与 estimator GB=10**9 一致
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


# 线性/混合注意力架构关键词 (config.json 的 model_type / architectures)
_LINEAR_ATTN_HINTS = (
    "next", "mamba", "rwkv", "retnet", "linear_attn", "lightning",
    "gated_delta", "hybrid", "minimax", "jamba", "hgrn",
)


def _detect_architecture(cfg: dict, spec: ModelSpec, model_id: str) -> None:
    """从 config.json 识别 MoE(激活参数量) 与 线性/混合注意力(KV 系数)。"""
    name_blob = (
        f"{model_id} {cfg.get('model_type','')} {' '.join(cfg.get('architectures') or [])}"
    ).lower()

    # ---- MoE ----
    n_exp = (
        cfg.get("num_experts")
        or cfg.get("n_routed_experts")
        or cfg.get("num_local_experts")
        or cfg.get("moe_num_experts")
        or 0
    )
    top_k = (
        cfg.get("num_experts_per_tok")
        or cfg.get("moe_top_k")
        or cfg.get("num_experts_per_token")
        or 0
    )
    if n_exp and n_exp > 1:
        spec.is_moe = True
        # 优先从模型名解析 "A3B" 这类激活参数量; 否则按 config 估算
        active = _active_b_from_name(model_id)
        if active is None and top_k:
            active = _estimate_active_params_b(cfg, spec, int(n_exp), int(top_k))
        if active:
            spec.active_params_b = round(active, 2)

    # ---- 线性/混合注意力 ----
    if any(h in name_blob for h in _LINEAR_ATTN_HINTS) or cfg.get("linear_attn_config"):
        spec.is_linear_attn = True
        # 混合架构常按比例混入少量全注意力层; 纯线性 KV≈0
        full_ratio = _full_attention_ratio(cfg)
        spec.kv_cache_factor = round(max(0.03, full_ratio), 3)


_ACTIVE_RE = re.compile(r"[aA](\d+(?:\.\d+)?)\s*[bB]\b")


def _active_b_from_name(model_id: str) -> Optional[float]:
    """从 '...-80B-A3B-...' 解析激活参数量 3.0(单位 B)。"""
    m = _ACTIVE_RE.search(model_id.replace("_", "-"))
    return float(m.group(1)) if m else None


def _full_attention_ratio(cfg: dict) -> float:
    """混合架构里全注意力层的占比; 拿不到则按 1/4 粗估。"""
    # 一些 config 用 layer_types / full_attention_interval 描述
    interval = cfg.get("full_attention_interval") or cfg.get("attn_interval")
    if isinstance(interval, int) and interval > 0:
        return 1.0 / interval
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if "full" in str(t).lower() or "attention" == str(t).lower())
        return full / len(layer_types) if full else 0.05
    return 0.25


def _estimate_active_params_b(cfg: dict, spec: ModelSpec, n_exp: int, top_k: int) -> Optional[float]:
    """按 config 粗估 MoE 每 token 激活参数量(B)。"""
    h = spec.hidden_size
    L = spec.num_layers
    moe_inter = (
        cfg.get("moe_intermediate_size")
        or cfg.get("intermediate_size")
        or 0
    )
    if not (h and L and moe_inter):
        return None
    shared = cfg.get("shared_expert_intermediate_size") or cfg.get("n_shared_experts") or 0
    if isinstance(shared, int) and shared and shared < 64:  # 当作"共享专家个数"
        shared_inter = moe_inter * shared
    else:
        shared_inter = shared if isinstance(shared, int) else 0
    # 每层: 注意力 ~ 4*h*h; 激活专家 MLP ~ top_k * 3 * h * moe_inter; 共享 ~ 3*h*shared_inter
    per_layer = 4 * h * h + (top_k * 3 * h * moe_inter) + (3 * h * shared_inter)
    embed = spec.vocab_size * h
    total_active = L * per_layer + embed
    return total_active / 1e9


# 官网搜索用的接口在前,SDK 列表接口兜底
_SEARCH_ENDPOINTS = (
    "https://modelscope.cn/api/v1/dolphin/models",
    f"{MS_BASE}/models",
)


async def search_models(query: str, limit: int = 10, page: int = 1) -> list[dict[str, Any]]:
    cache_key = f"search:{query}:{limit}:{page}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    body = {
        "Name": query,
        "PageSize": limit,
        "PageNumber": page,
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
    """Fetch config.json + model metadata from a model repo and map to ModelSpec fields."""
    cache_key = f"config:{model_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    spec = ModelSpec(model_id=model_id, params_b=params_from_name(model_id) or 0.0)
    spec.precision = precision_from_name(model_id)
    config_found = False
    name_params = spec.params_b  # from regex

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
            import asyncio

            api_url = f"{MS_BASE}/models/{model_id}"

            async def fetch_cfg():
                url = f"{MS_BASE}/models/{model_id}/repo?Revision=master&FilePath=config.json"
                try:
                    r = await client.get(url)
                    if r.status_code != 200:
                        return {"__error": f"status {r.status_code}"}
                    data = r.json()
                except Exception as exc:
                    return {"__error": str(exc)}

                # 递归/迭代解包，直到找到 config dict（含 hidden_size 或 text_config）
                candidates = [data]
                seen = set()
                while candidates:
                    cur = candidates.pop(0)
                    cid = id(cur)
                    if cid in seen:
                        continue
                    seen.add(cid)

                    if isinstance(cur, dict):
                        # 找到目标 dict
                        if "hidden_size" in cur or "text_config" in cur or "num_hidden_layers" in cur:
                            return cur
                        # 把可能包装字段加入候选
                        for k in ("Data", "data", "Content", "content", "text", "result", "Result"):
                            if k in cur:
                                candidates.append(cur[k])
                    elif isinstance(cur, str) and cur.strip().startswith("{"):
                        try:
                            candidates.append(json.loads(cur))
                        except Exception:
                            pass
                    elif isinstance(cur, list) and cur:
                        candidates.extend(cur)

                return {"__error": "no hidden_size found", "preview": str(data)[:300]}

            async def fetch_meta():
                r = await client.get(api_url)
                if r.status_code != 200:
                    return (None, None, None)
                data = r.json()
                dd = (data.get("Data") or data) if isinstance(data, dict) else {}
                # 权重大小
                s = dd.get("StorageSize") or dd.get("storageSize")
                wgb = round(s / 1e9, 2) if isinstance(s, (int, float)) and s > 0 else None
                # 参数量: ReadMeContent 中的 "XB in total"
                txt = dd.get("ReadMeContent") or dd.get("Description") or ""
                m = re.search(r'(\d+(?:\.\d+)?)\s*[bB]\s+in\s+total', txt)
                p = float(m.group(1)) if m else None
                return wgb, p, dd

            cfg, (wgb, html_params, _) = await asyncio.gather(fetch_cfg(), fetch_meta())
            cfg_error = cfg.get("__error") if isinstance(cfg, dict) else str(cfg)

            # 处理 config.json (支持顶层或 text_config 嵌套)
            if cfg and isinstance(cfg, dict) and not cfg_error:
                config_found = True
                tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

                def _c(key: str):
                    """优先顶层，其次 text_config。"""
                    return cfg.get(key, tc.get(key))

                if _c("hidden_size") is not None:
                    spec.hidden_size = int(_c("hidden_size"))
                if _c("num_hidden_layers") is not None:
                    spec.num_layers = int(_c("num_hidden_layers"))
                if _c("num_attention_heads") is not None:
                    spec.num_attention_heads = int(_c("num_attention_heads"))
                if _c("num_key_value_heads") is not None:
                    spec.num_key_value_heads = int(_c("num_key_value_heads"))
                if _c("vocab_size") is not None:
                    spec.vocab_size = int(_c("vocab_size"))

                # 混合注意力 (Gemma4 等)
                sw = _c("sliding_window")
                if sw is not None:
                    spec.sliding_window = int(sw)
                layer_types = _c("layer_types")
                if isinstance(layer_types, list):
                    spec.num_full_attention_layers = layer_types.count("full_attention")
                ngkv = _c("num_global_key_value_heads")
                if ngkv is not None:
                    spec.num_global_key_value_heads = int(ngkv)
                ghd = _c("global_head_dim")
                if ghd is not None:
                    spec.global_head_dim = int(ghd)
                # 如果 text_config 里直接给了 head_dim，用它来校正 head_dim
                hd = _c("head_dim")
                if hd is not None:
                    spec.head_dim = int(hd)

                td = cfg.get("torch_dtype") or tc.get("torch_dtype") or cfg.get("dtype") or tc.get("dtype")
                if td:
                    spec.precision = str(td).replace("bfloat", "BF").replace("float", "FP").upper()
                spec.tensor_types = _extract_tensor_types(cfg, model_id)

                # 架构特征: MoE / 线性注意力 (影响吞吐建模)
                _detect_architecture(cfg, spec, model_id)

            # 处理权重
            if wgb:
                spec.weight_size_gb = wgb

            # 处理参数量
            if html_params:
                spec.params_b = html_params
                spec.params_accurate = True
            elif name_params > 0:
                spec.params_accurate = True
            elif wgb:
                inferred = params_from_storage(wgb * 1e9, model_id)
                if inferred:
                    spec.params_b = inferred
                    spec.params_accurate = False
    except Exception:
        config_found = False

    out = spec.model_dump()
    out["config_found"] = config_found
    if not config_found and cfg_error:
        out["config_error"] = cfg_error
    _cache_set(cache_key, out)
    return out


def _extract_tensor_types(cfg: dict, model_id: str) -> str:
    """从 config.json 提取完整的张量类型信息，格式类似 'INT8, E8M0, BF16, F32'。"""
    parts: list[str] = []

    # 1. quantization_config.quant_method
    qt = cfg.get("quantization_config")
    if isinstance(qt, dict):
        qm = qt.get("quant_method", "").lower()
        fmt = qt.get("fmt", "")
        scale = qt.get("scale_fmt", "")
        if qm == "fp8":
            parts.append(f"FP8_{fmt.upper()}" if fmt else "FP8")
        elif qm:
            parts.append(qm.upper())
        if scale:
            parts.append(scale.upper())

    # 2. expert_dtype (MoE)
    ed = cfg.get("expert_dtype")
    if ed:
        parts.append(ed.upper())

    # 3. torch_dtype
    td = cfg.get("torch_dtype")
    if td:
        dt = str(td).replace("bfloat", "BF").replace("float", "FP").upper()
        if dt not in parts:
            parts.append(dt)

    # 4. 额外: 某些模型用 F32 分量
    if cfg.get("expert_dtype") == "fp4":
        if "FP4" not in parts:
            parts.append("FP4")

    if parts:
        return ", ".join(parts)
    return precision_from_name(model_id)


async def _fetch_model_meta(client: httpx.AsyncClient, model_id: str) -> Optional[dict]:
    """从 ModelScope 页面和 API 获取参数量和 StorageSize。"""
    try:
        result: dict = {}

        # 1. API 获取 StorageSize
        api_url = f"{MS_BASE}/models/{model_id}"
        api_resp = await client.get(api_url)
        if api_resp.status_code == 200:
            data = api_resp.json()
            if isinstance(data, dict):
                d = data.get("Data") or data
                if isinstance(d, dict):
                    storage = d.get("StorageSize") or d.get("storageSize")
                    if isinstance(storage, (int, float)) and storage > 0:
                        result["weight_gb"] = round(storage / 1e9, 2)

        # 2. HTML 页面提取参数量 (API 无此字段)
        page_url = f"https://modelscope.cn/models/{model_id}"
        html_headers = {**_HEADERS, "Accept": "text/html,application/xhtml+xml"}
        page_resp = await client.get(page_url, headers=html_headers)
        if page_resp.status_code == 200:
            html = page_resp.text
            # 匹配页面中的参数量显示: 80B / 80.0B / 7.2B 等
            import re
            m = re.search(r'(\d+(?:\.\d+)?)\s*[bB]', html)
            if m:
                try:
                    result["params"] = float(m.group(1))
                except ValueError:
                    pass

        return result if result else None
    except Exception:
        pass
    return None


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
