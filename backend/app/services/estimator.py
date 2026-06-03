"""Inference memory & throughput estimation engine.

All formulas are documented inline and cross-referenced to the PRD (§4).
The engine is intentionally pure (no I/O) so it can be unit-tested and, if
needed, mirrored on the frontend as an offline fallback.

Coordinate system / assumptions
--------------------------------
* All GPUs are treated as one tensor-parallel (TP) group of size = total card
  count. Weights and KV cache are sharded evenly across the group; aggregate
  memory bandwidth and FLOPS therefore add up across cards.
* Decode is modelled as memory-bandwidth bound (read all weights + KV per
  step); a single weight read serves the whole in-flight batch, which is why
  batching boosts TPS until the compute roofline is hit.
* Prefill is modelled as compute bound (2 * P FLOPs per token).
"""
from __future__ import annotations

import math

from ..models.schemas import (
    EstimateRequest,
    EstimateResponse,
    InferenceConfig,
    MemoryBreakdown,
    ModelSpec,
)

GB = 1024**3

# bytes per parameter for a given quantisation of the *weights*
QUANT_BYTES: dict[str, float] = {
    "FP16": 2.0,
    "BF16": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "INT4": 0.5,
    "GPTQ": 0.5,
    "AWQ": 0.5,
}

# extra multiplier for group-quantised formats (scales / zero-points)
QUANT_OVERHEAD: dict[str, float] = {
    "GPTQ": 1.10,
    "AWQ": 1.08,
    "INT4": 1.05,
}

KV_BYTES: dict[str, float] = {"FP16": 2.0, "BF16": 2.0, "FP8": 1.0, "INT8": 1.0}

# compute-utilisation (η) midpoints per framework — PRD §4.3
FRAMEWORK_COMPUTE_UTIL: dict[str, float] = {
    "TensorRT-LLM": 0.72,
    "vLLM": 0.60,
    "SGLang": 0.65,
    "llama.cpp": 0.45,
}

# achievable fraction of peak HBM bandwidth during decode, per framework
FRAMEWORK_MEM_UTIL: dict[str, float] = {
    "TensorRT-LLM": 0.85,
    "vLLM": 0.80,
    "SGLang": 0.80,
    "llama.cpp": 0.60,
}

# PagedAttention KV-cache packing efficiency (vLLM/SGLang waste a little)
KV_UTIL = 0.90

# activation memory empirical coefficient (PRD §4.1, α is per-token, per-layer
# working set is NOT all materialised at once during inference, so we use a
# modest peak-activation model rather than the literal L-product).
ACT_ALPHA = 2.0

# fixed framework + CUDA context overhead per GPU (bytes)
OVERHEAD_PER_GPU = 1.2 * GB


def _quant_bytes(quant: str) -> float:
    return QUANT_BYTES.get(quant, 2.0)


def _kv_heads(model: ModelSpec) -> int:
    return model.num_key_value_heads or model.num_attention_heads


def estimate_memory(req: EstimateRequest) -> tuple[MemoryBreakdown, list[str]]:
    model = req.model
    inf = req.inference
    warnings: list[str] = []

    n_gpu = sum(g.count for g in req.gpus)
    if n_gpu <= 0:
        n_gpu = 1

    params = model.params_b * 1e9
    head_dim = model.hidden_size / max(1, model.num_attention_heads)
    kv_dim = _kv_heads(model) * head_dim  # GQA-aware KV width

    seq = inf.input_len + inf.output_len
    # active sequences in flight = concurrency, each padded to batch grouping
    active_seqs = inf.concurrency

    # --- weights (PRD §4.1) ---
    wb = _quant_bytes(inf.quant)
    weights = params * wb * QUANT_OVERHEAD.get(inf.quant, 1.0)

    # --- KV cache (PRD §4.1, GQA-aware) ---
    kv_b = KV_BYTES.get(inf.kv_quant, 2.0)
    kv = 2 * model.num_layers * kv_dim * seq * active_seqs * kv_b
    if inf.framework in ("vLLM", "SGLang"):
        kv = kv / KV_UTIL  # account for paging overhead -> larger reservation

    # --- activations (modest peak model) ---
    act = active_seqs * seq * model.hidden_size * wb * ACT_ALPHA

    overhead = OVERHEAD_PER_GPU * n_gpu

    total = weights + kv + act + overhead
    # TP shards weights+kv across all cards; activations + overhead are roughly
    # per-card resident, so per-gpu = sharded part / n + per-card part.
    per_gpu = (weights + kv) / n_gpu + act / n_gpu + OVERHEAD_PER_GPU

    breakdown = MemoryBreakdown(
        weights_gb=weights / GB,
        kv_cache_gb=kv / GB,
        activations_gb=act / GB,
        overhead_gb=overhead / GB,
        total_gb=total / GB,
        per_gpu_gb=per_gpu / GB,
    )
    return breakdown, warnings


def estimate(req: EstimateRequest) -> EstimateResponse:
    model = req.model
    inf = req.inference
    mem, warnings = estimate_memory(req)

    n_gpu = sum(g.count for g in req.gpus) or 1
    params = model.params_b * 1e9

    # aggregate hardware roofline across the whole TP group
    use_fp8 = inf.quant in ("FP8",)
    flops_total = 0.0
    bw_total = 0.0  # bytes/s
    total_mem_gb = 0.0
    all_nvlink = True
    for g in req.gpus:
        peak = g.spec.fp8_tflops if use_fp8 and g.spec.fp8_tflops > 0 else g.spec.fp16_tflops
        if use_fp8 and g.spec.fp8_tflops <= 0:
            warnings.append(
                f"{g.spec.name} 无原生 FP8 算力,已回退到 FP16 算力估算。"
            )
        flops_total += peak * 1e12 * g.count
        bw_total += g.spec.bw_gbs * 1e9 * g.count
        total_mem_gb += g.spec.mem_gb * g.count
        all_nvlink = all_nvlink and g.spec.nvlink

    if flops_total <= 0:
        flops_total = 1e12
        warnings.append("所选 GPU 算力为 0(占位/未知),TPS 仅供参考。")

    compute_util = (
        inf.compute_util
        if inf.compute_util is not None
        else FRAMEWORK_COMPUTE_UTIL.get(inf.framework, 0.6)
    )
    mem_util = (
        inf.mem_util
        if inf.mem_util is not None
        else FRAMEWORK_MEM_UTIL.get(inf.framework, 0.8)
    )

    # interconnect derate for TP all-reduce when no high-speed link
    if n_gpu > 1 and not all_nvlink:
        compute_util *= 0.85
        warnings.append("存在无高速互联(NVLink/HCCS)的卡,TP 通信开销已下调效率 ~15%。")

    batch = inf.concurrency  # continuous batching: in-flight requests

    # ---------------- PREFILL (compute bound) — PRD §4.2 ----------------
    # 2*P FLOPs per token, all input tokens of all in-flight requests.
    prefill_flops = 2 * params * inf.input_len * batch
    t_prefill = prefill_flops / (flops_total * compute_util)  # seconds
    ttft = t_prefill / max(1, batch)  # per-request first-token latency

    # ---------------- DECODE (bandwidth bound) — PRD §4.2 ----------------
    # bytes moved per decode step: full weights (sharded across TP, but
    # aggregate BW also scales, so use totals) + KV read for active batch.
    weights_bytes = mem.weights_gb * GB
    kv_bytes = mem.kv_cache_gb * GB
    bytes_per_step = weights_bytes + kv_bytes
    tpot = bytes_per_step / (bw_total * mem_util)  # seconds / token (whole batch)

    # compute floor for decode (rarely binding, but caps tiny-model TPS)
    decode_compute_t = (2 * params * batch) / (flops_total * compute_util)
    tpot_eff = max(tpot, decode_compute_t)

    decode_tps = batch / tpot_eff if tpot_eff > 0 else 0.0

    # ---------------- combine ----------------
    t_decode = inf.output_len * tpot_eff
    request_latency = ttft + t_decode  # seconds, per request

    total_tokens = (inf.input_len + inf.output_len) * batch
    total_time = t_prefill + t_decode
    tps = total_tokens / total_time if total_time > 0 else 0.0

    # ---------------- bottleneck analysis ----------------
    bottleneck, suggestions = _analyse(
        mem=mem,
        total_mem_gb=total_mem_gb,
        n_gpu=n_gpu,
        tpot=tpot,
        decode_compute_t=decode_compute_t,
        t_prefill=t_prefill,
        t_decode=t_decode,
        inf=inf,
    )

    util = mem.total_gb / total_mem_gb if total_mem_gb > 0 else 0.0
    fits = mem.per_gpu_gb <= (total_mem_gb / n_gpu)

    if not fits:
        suggestions.insert(
            0,
            "⚠ 单卡显存超限!建议:降低量化精度 / 减小并发或上下文 / 增加卡数(提高 TP)。",
        )

    return EstimateResponse(
        memory=mem,
        total_mem_gb=total_mem_gb,
        mem_utilization=util,
        fits=fits,
        tps=tps,
        ttft_ms=ttft * 1000,
        tpot_ms=tpot_eff * 1000,
        request_latency_ms=request_latency * 1000,
        bottleneck=bottleneck,
        suggestions=suggestions,
        effective_compute_util=compute_util,
        effective_mem_util=mem_util,
        warnings=warnings,
    )


def _analyse(
    *,
    mem: MemoryBreakdown,
    total_mem_gb: float,
    n_gpu: int,
    tpot: float,
    decode_compute_t: float,
    t_prefill: float,
    t_decode: float,
    inf: InferenceConfig,
) -> tuple[str, list[str]]:
    suggestions: list[str] = []

    util = mem.total_gb / total_mem_gb if total_mem_gb > 0 else 0.0

    # memory pressure wins if we're near capacity
    if util > 0.92:
        suggestions.append("显存接近上限,建议降低量化精度(如 FP16→INT8)或减小并发/上下文。")
        return "Memory Bound", suggestions

    # decode is bandwidth bound if BW time dominates the compute floor
    if tpot >= decode_compute_t and t_decode >= t_prefill * 0.5:
        suggestions.append("Decode 阶段受显存带宽限制,选用更高带宽显卡(如 H20/H200)可直接提升 TPS。")
        suggestions.append("提高并发可摊薄权重读取成本,显著提升总 TPS(直到触及算力上限)。")
        return "Bandwidth Bound", suggestions

    # otherwise compute bound (long prefill / heavy batch)
    suggestions.append("受算力限制,建议增加 GPU 数量或换用更高算力显卡(如 H100/H200)。")
    if inf.input_len > 8 * inf.output_len:
        suggestions.append("输入远长于输出,Prefill 占主导;可考虑 Prefill/Decode 分离部署。")
    return "Compute Bound", suggestions
