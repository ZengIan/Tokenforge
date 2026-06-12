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
    # 参数量: 优先从 safetensors.model_size 获取 (精确参数量), 其次从模型名正则
    params_b = params_from_name(model_id)
    model_infos = m.get("ModelInfos") if isinstance(m.get("ModelInfos"), dict) else None
    if model_infos:
        st = model_infos.get("safetensor") if isinstance(model_infos.get("safetensor"), dict) else None
        if st:
            ms = st.get("model_size")
            if isinstance(ms, (int, float)) and ms > 0:
                params_b = round(ms / 1e9, 2)
    return {
        "model_id": model_id,
        "name": name,
        "chinese_name": m.get("ChineseName") or m.get("chineseName") or name,
        "params_b": params_b,
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

    # ---- 注意力类型 (MHA/GQA/MQA/MLA), 对 KV cache 影响极大 ----
    tc_inner = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    kv_lora = cfg.get("kv_lora_rank", tc_inner.get("kv_lora_rank"))
    if kv_lora and int(kv_lora) > 0:
        # MLA(DeepSeek-V2/V3、GLM-MLA/GlmMoeDSA 等): 低秩潜在 KV, 与头数解耦
        rope = int(cfg.get("qk_rope_head_dim", tc_inner.get("qk_rope_head_dim")) or 64)
        spec.attn_type = "MLA"
        spec.mla_kv_dim = int(kv_lora) + rope
    else:
        kvh = spec.num_key_value_heads or spec.num_attention_heads
        ah = spec.num_attention_heads
        if kvh <= 1:
            spec.attn_type = "MQA"
        elif kvh < ah:
            spec.attn_type = "GQA"
        else:
            spec.attn_type = "MHA"

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
                    return (None, None)
                data = r.json()
                dd = (data.get("Data") or data) if isinstance(data, dict) else {}
                # 权重大小
                s = dd.get("StorageSize") or dd.get("storageSize")
                wgb = round(s / 1e9, 2) if isinstance(s, (int, float)) and s > 0 else None
                # 参数量: safetensors 元数据中的 model_size (精确参数量)
                st_params = None
                model_infos = dd.get("ModelInfos") if isinstance(dd.get("ModelInfos"), dict) else None
                if model_infos:
                    st = model_infos.get("safetensor") if isinstance(model_infos.get("safetensor"), dict) else None
                    if st:
                        ms = st.get("model_size")
                        if isinstance(ms, (int, float)) and ms > 0:
                            st_params = round(ms / 1e9, 2)
                return wgb, st_params

            cfg, (wgb, st_params) = await asyncio.gather(fetch_cfg(), fetch_meta())
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
                # 合并 cfg + text_config, 使 _detect_architecture 能查到嵌套字段
                # 注意: cfg 顶层可能含 None 值会覆盖 tc 的有效值, 需过滤
                merged_cfg = {**tc, **{k: v for k, v in cfg.items() if v is not None}}
                _detect_architecture(merged_cfg, spec, model_id)

            # 处理权重
            if wgb:
                spec.weight_size_gb = wgb

            # 处理参数量: 优先 safetensors.model_size > 模型名正则;
            # 都没有则保持为 0, 由用户手动填。
            if st_params and st_params > 0:
                spec.params_b = st_params
                spec.params_accurate = True
            elif name_params > 0:
                spec.params_accurate = True
    except Exception:
        config_found = False
        cfg_error = "config fetch exception"

    # 降级: 即使 config.json 拉取失败或未识别 MoE，也从模型名中检测
    if not spec.is_moe:
        active = _active_b_from_name(model_id)
        if active:
            spec.is_moe = True
            spec.active_params_b = round(active, 2)

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
