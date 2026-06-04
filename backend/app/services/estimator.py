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

from ..models.schemas import (
    EstimateRequest,
    EstimateResponse,
    InferenceConfig,
    MemoryBreakdown,
    ModelSpec,
)

GB = 10**9

# --dtype -> 每元素字节数 (auto 视为 BF16)
DTYPE_BYTES: dict[str, float] = {
    "auto": 2.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "float32": 4.0,
}

# --quantization -> 每参数权重字节数。"none" 表示不量化,按 --dtype 计。
QUANT_BYTES: dict[str, float] = {
    "fp8": 1.0,
    "int8": 1.0,
    "w8a8": 1.0,
    "awq": 0.5,     # 4bit 组量化
    "gptq": 0.5,    # 4bit 组量化
    "w4a8": 0.5,
    "w4a16": 0.5,
}

# 组量化的 scale/zero-point 额外开销
QUANT_OVERHEAD: dict[str, float] = {
    "awq": 1.08,
    "gptq": 1.10,
    "w4a8": 1.05,
    "w4a16": 1.05,
}

# 激活为 8bit/FP8 的量化方案,decode 计算可走低精度 Tensor Core
LOW_PRECISION_COMPUTE = {"fp8", "w8a8", "w4a8"}

# --kv-cache-dtype -> 每元素字节数。auto 跟随 --dtype。
KV_DTYPE_BYTES: dict[str, float] = {
    "fp8": 1.0,
    "fp8_e5m2": 1.0,
    "fp8_e4m3": 1.0,
    "int8": 1.0,
}

# vLLM 的算力/带宽利用率经验系数
VLLM_COMPUTE_UTIL = 0.60
VLLM_MEM_UTIL = 0.80

# PagedAttention KV-cache packing efficiency
KV_UTIL = 0.90

# activation memory empirical coefficient (peak working set ~ batched tokens)
ACT_ALPHA = 2.0

# fixed framework + CUDA context overhead per GPU (bytes)
OVERHEAD_PER_GPU = 1.2 * GB

# ---- 单 token 固定开销模型(decode 小 batch 的现实地板)----
# 屋顶线只算"权重/带宽时间",但 batch=1 时真正耗时常被 kernel 启动、TP all-reduce
# 延迟、采样等固定开销主导(尤其 enforce_eager 关掉 CUDA Graph 时)。下列系数按
# H20 + vLLM 实测点标定: Gemma 60层 dense eager TP2 ≈ 真实 <20 tok/s;
# Qwen3-Next 48层 MoE+线性 eager TP4 ≈ 真实 ~9 tok/s。
PER_LAYER_OVERHEAD_MS = 0.60       # 每层每 token 基础开销(eager 口径)
CUDA_GRAPH_FACTOR = 0.12           # 开启 CUDA Graph(未 enforce_eager)开销大幅降低
MOE_OVERHEAD_MULT = 1.8            # MoE 的 expert gather/scatter 额外开销
LINEAR_ATTN_OVERHEAD_MULT = 1.5    # 线性/混合注意力的逐 token 串行递归额外开销
# 区间(相对中值): 乐观/保守 的开销系数
OVERHEAD_FAST = 0.70
OVERHEAD_SLOW = 1.60


def _weight_bytes(inf: InferenceConfig) -> float:
    """运行时每参数权重字节数:有量化按量化,否则按 dtype。"""
    if inf.quantization != "none":
        return QUANT_BYTES.get(inf.quantization, 2.0)
    return DTYPE_BYTES.get(inf.dtype, 2.0)


def _act_bytes(inf: InferenceConfig) -> float:
    return DTYPE_BYTES.get(inf.dtype, 2.0)


def _kv_bytes(inf: InferenceConfig) -> float:
    """KV 缓存每元素字节数;auto 跟随 dtype。"""
    if inf.kv_cache_dtype != "auto":
        return KV_DTYPE_BYTES.get(inf.kv_cache_dtype, 2.0)
    return DTYPE_BYTES.get(inf.dtype, 2.0)


def _kv_heads(model: ModelSpec) -> int:
    return model.num_key_value_heads or model.num_attention_heads


def estimate_memory(req: EstimateRequest) -> tuple[MemoryBreakdown, list[str]]:
    model = req.model
    inf = req.inference
    warnings: list[str] = []

    n_gpu = sum(g.count for g in req.gpus)
    if n_gpu <= 0:
        n_gpu = 1

    head_dim = model.hidden_size / max(1, model.num_attention_heads)
    kv_dim = _kv_heads(model) * head_dim  # GQA-aware KV width

    # KV cache 按 --max-model-len 每路预留, --max-num-seqs 路并发。
    seq = inf.max_model_len
    active_seqs = inf.max_num_seqs

    # --- weights (运行时显存, 量化感知) ---
    # 量化时按运行精度算(官方文件大小可能是另一精度); 不量化且有官方大小则用官方值。
    if inf.quantization == "none" and model.weight_size_gb and model.weight_size_gb > 0:
        weights = model.weight_size_gb * GB
    else:
        weights = model.params_b * 1e9 * _weight_bytes(inf)
    weights *= QUANT_OVERHEAD.get(inf.quantization, 1.0)

    # --- KV cache (GQA-aware; 线性/混合注意力按 kv_cache_factor 缩减) ---
    kv_b = _kv_bytes(inf)
    kv = 2 * model.num_layers * kv_dim * seq * active_seqs * kv_b * model.kv_cache_factor
    kv = kv / KV_UTIL  # PagedAttention 分页预留略大于裸算

    # --- activations: 峰值 ~ 单次迭代批处理 token 数 (--max-num-batched-tokens) ---
    act = inf.max_num_batched_tokens * model.hidden_size * _act_bytes(inf) * ACT_ALPHA

    # --enforce-eager 关闭 CUDA Graph,省去其捕获显存(~0.6GB/卡)
    overhead_per_gpu = OVERHEAD_PER_GPU - (0.6 * GB if inf.enforce_eager else 0)
    overhead = overhead_per_gpu * n_gpu

    total = weights + kv + act + overhead
    # TP shards weights+kv across all cards; activations + overhead are roughly
    # per-card resident, so per-gpu = sharded part / n + per-card part.
    per_gpu = (weights + kv) / n_gpu + act / n_gpu + overhead_per_gpu

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
    use_fp8 = inf.quantization in LOW_PRECISION_COMPUTE
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

    compute_util = inf.compute_util if inf.compute_util is not None else VLLM_COMPUTE_UTIL
    mem_util = inf.mem_util if inf.mem_util is not None else VLLM_MEM_UTIL

    # interconnect derate for TP all-reduce when no high-speed link
    if n_gpu > 1 and not all_nvlink:
        compute_util *= 0.85
        warnings.append("无高速互联(NVLink/HCCS),TP 通信开销已下调效率 ~15%。")

    batch = inf.max_num_seqs  # continuous batching: in-flight sequences

    # MoE: decode/prefill 只用激活参数(全部专家常驻显存, 但每 token 只读 top-k)
    active_params = (model.active_params_b or model.params_b) * 1e9
    active_ratio = active_params / params if params > 0 else 1.0

    # ---------------- PREFILL (compute bound) ----------------
    prefill_flops = 2 * active_params * inf.max_num_batched_tokens
    ttft = prefill_flops / (flops_total * compute_util)  # seconds

    # ---------------- DECODE 屋顶线 ----------------
    # 每步只读激活权重(KV 读取量随上下文变化, 这里并入下方固定开销, 不重复计)。
    weight_read_bytes = mem.weights_gb * GB * active_ratio
    tpot_bw = weight_read_bytes / (bw_total * mem_util)  # 带宽时间
    decode_compute_t = (2 * active_params * batch) / (flops_total * compute_util)
    tpot_roof = max(tpot_bw, decode_compute_t)  # 理论屋顶线(每步)

    # ---------------- 每 token 固定开销(现实地板) ----------------
    arch_mult = 1.0
    if model.is_moe:
        arch_mult *= MOE_OVERHEAD_MULT
    if model.is_linear_attn:
        arch_mult *= LINEAR_ATTN_OVERHEAD_MULT
    tp_mult = 1.0 + 0.15 * (n_gpu - 1)
    graph_mult = 1.0 if inf.enforce_eager else CUDA_GRAPH_FACTOR
    overhead_s = (
        model.num_layers * PER_LAYER_OVERHEAD_MS * arch_mult * tp_mult * graph_mult
    ) / 1000.0

    # TPOT = 屋顶线 + 固定开销; 区间由开销不确定性给出
    tpot_mid = tpot_roof + overhead_s
    tpot_fast = tpot_roof + overhead_s * OVERHEAD_FAST
    tpot_slow = tpot_roof + overhead_s * OVERHEAD_SLOW
    tpot_eff = tpot_mid

    def _safe(x: float) -> float:
        return 1.0 / x if x > 0 else 0.0

    single_tps = _safe(tpot_mid)
    single_tps_high = _safe(tpot_fast)
    single_tps_low = _safe(tpot_slow)
    tps = batch * single_tps
    tps_high = batch * single_tps_high
    tps_low = batch * single_tps_low
    request_latency = ttft + tpot_mid  # 首字 + 单 token
    # 供瓶颈分析判断带宽 vs 算力
    tpot = tpot_bw

    # --gpu-memory-utilization 限定可用显存预算
    per_gpu_capacity = (total_mem_gb / n_gpu) * inf.gpu_memory_utilization
    budget = per_gpu_capacity * n_gpu
    util = mem.total_gb / budget if budget > 0 else 0.0
    fits = mem.per_gpu_gb <= per_gpu_capacity

    # 显存预算实际可容纳的并发序列数
    non_kv_gb = mem.total_gb - mem.kv_cache_gb  # 权重+激活+开销
    kv_per_seq_gb = mem.kv_cache_gb / batch if batch > 0 else 0.0
    avail_kv_gb = budget - non_kv_gb
    if kv_per_seq_gb > 0:
        max_fit_seqs = max(0, int(avail_kv_gb / kv_per_seq_gb))
    else:
        max_fit_seqs = batch

    # ---------------- bottleneck analysis ----------------
    bottleneck, suggestions = _analyse(
        mem=mem,
        total_mem_gb=total_mem_gb,
        n_gpu=n_gpu,
        tpot=tpot,
        decode_compute_t=decode_compute_t,
        inf=inf,
    )

    if not fits:
        suggestions.insert(
            0,
            f"⚠ 超出显存预算!当前配置约可容纳 {max_fit_seqs} 路并发(< max-num-seqs={batch})。"
            "建议:量化权重(awq/fp8) / 调小 max-model-len 或 max-num-seqs / "
            "用 fp8 kv-cache / 增加卡数 / 调高 --gpu-memory-utilization。",
        )
    elif max_fit_seqs < batch:
        suggestions.append(
            f"max-num-seqs={batch} 超过显存可容纳的约 {max_fit_seqs} 路,"
            "vLLM 实际并发会收敛到该值附近(排队)。"
        )

    # 性能/瓶颈分析的可靠性: 依赖算力&带宽规格。占位估值(source=estimate)或
    # 算力/带宽为 0 时, TPS/延迟/瓶颈判定不可信(显存与可容纳并发仍可参考)。
    analysis_reliable = all(
        (g.spec.source == "datasheet" or g.spec.source.startswith("measured"))
        and g.spec.fp16_tflops > 0
        and g.spec.bw_gbs > 0
        for g in req.gpus
    )
    if not analysis_reliable:
        suggestions.insert(
            0,
            "⚠ 所选 GPU 的算力/带宽为占位估值(source=estimate)或未知,"
            "TPS、延迟与瓶颈判定不可靠;显存占用与可容纳并发仍可参考。"
            "请在 gpus.yaml 填入厂商规格书或实测值(source 改为 datasheet/measured)后再看性能结论。",
        )

    return EstimateResponse(
        memory=mem,
        total_mem_gb=total_mem_gb,
        mem_utilization=util,
        fits=fits,
        tps=tps,
        tps_low=tps_low,
        tps_high=tps_high,
        single_tps=single_tps,
        single_tps_low=single_tps_low,
        single_tps_high=single_tps_high,
        ttft_ms=ttft * 1000,
        tpot_ms=tpot_eff * 1000,
        request_latency_ms=request_latency * 1000,
        max_fit_seqs=max_fit_seqs,
        bottleneck=bottleneck,
        suggestions=suggestions,
        analysis_reliable=analysis_reliable,
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
    inf: InferenceConfig,
) -> tuple[str, list[str]]:
    suggestions: list[str] = []

    util = mem.total_gb / total_mem_gb if total_mem_gb > 0 else 0.0

    # memory pressure wins if we're near capacity
    if util > 0.92:
        suggestions.append("显存接近上限,建议量化权重(awq/fp8)或用 fp8 kv-cache、减小并发/上下文。")
        return "Memory Bound", suggestions

    # decode is bandwidth bound if BW time dominates the compute floor
    if tpot >= decode_compute_t:
        suggestions.append("Decode 受显存带宽限制,选用更高带宽显卡(如 H20/H200)可直接提升 TPS。")
        suggestions.append("提高 max-num-seqs 可摊薄权重读取成本,显著提升总 TPS(直到触及算力上限)。")
        return "Bandwidth Bound", suggestions

    # otherwise compute bound (heavy batch)
    suggestions.append("受算力限制,建议增加 GPU 数量或换用更高算力显卡(如 H100/H200)。")
    return "Compute Bound", suggestions
