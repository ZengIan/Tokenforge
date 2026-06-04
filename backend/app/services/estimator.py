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

# 单机最多卡数(超过即视为多机/跨机部署)
GPUS_PER_NODE = 8
# 跨机互联对算力效率的折扣(乘到 compute_util)
INTERNODE_COMPUTE_FACTOR = {
    "nvlink": 1.00,   # NVLink Switch/HCCS 等高速无损 -> 接近无损
    "ib": 0.90,       # InfiniBand/RoCE 高速网 -> ~10% 损耗
    "ethernet": 0.75, # 普通以太网 -> ~25% 损耗
}
# 跨机互联对每 token 固定开销的放大(跨机 all-reduce/PP 通信延迟更高)
INTERNODE_OVERHEAD_MULT = {
    "nvlink": 1.0,
    "ib": 1.4,
    "ethernet": 2.2,
}

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
    if model.num_key_value_heads is not None and model.num_key_value_heads > 0:
        return model.num_key_value_heads
    return model.num_attention_heads


def estimate_memory(req: EstimateRequest) -> tuple[MemoryBreakdown, list[str]]:
    model = req.model
    inf = req.inference
    warnings: list[str] = []

    n_gpu = sum(g.count for g in req.gpus)
    if n_gpu <= 0:
        n_gpu = 1

    total_mem_gb = sum(g.spec.mem_gb * g.count for g in req.gpus)
    budget = total_mem_gb * inf.gpu_memory_utilization

    # head_dim: 优先用 config.json 直接提供的，否则推导
    if model.head_dim is not None and model.head_dim > 0:
        head_dim = float(model.head_dim)
    else:
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

    # --- activations: 峰值 ~ 单次迭代批处理 token 数 (--max-num-batched-tokens) ---
    act = inf.max_num_batched_tokens * model.hidden_size * _act_bytes(inf) * ACT_ALPHA

    # --enforce-eager 关闭 CUDA Graph,省去其捕获显存(~0.6GB/卡)
    overhead_per_gpu = OVERHEAD_PER_GPU - (0.6 * GB if inf.enforce_eager else 0)
    overhead = overhead_per_gpu * n_gpu

    # --- KV cache: 先算理论需求, 再按实际显存预算截断 ---
    kv_b = _kv_bytes(inf)

    if model.sliding_window and model.num_full_attention_layers > 0:
        # 混合注意力 (如 Gemma4): sliding 层只缓存 window 内 token, full 层缓存完整 seq
        n_full = model.num_full_attention_layers
        n_slide = model.num_layers - n_full

        # sliding attention layers
        kv_slide = 2 * n_slide * kv_dim * model.sliding_window * active_seqs * kv_b

        # full attention layers (可能用不同的 KV 配置)
        if model.num_global_key_value_heads and model.global_head_dim:
            kv_dim_full = model.num_global_key_value_heads * model.global_head_dim
        else:
            kv_dim_full = kv_dim
        kv_full = 2 * n_full * kv_dim_full * seq * active_seqs * kv_b

        kv_theory = (kv_slide + kv_full) / KV_UTIL * model.kv_cache_factor
        # 每路平均 KV (用于反推可容纳并发)
        kv_per_seq = (kv_slide + kv_full) / KV_UTIL * model.kv_cache_factor / active_seqs if active_seqs > 0 else 0.0
    else:
        # 标准模型: 所有层缓存完整上下文 (线性/混合注意力按 kv_cache_factor 缩减)
        kv_per_seq = 2 * model.num_layers * kv_dim * seq * kv_b * model.kv_cache_factor / KV_UTIL
        kv_theory = kv_per_seq * active_seqs

    non_kv = weights + act + overhead
    # 总预算 = budget * GB (bytes)
    kv_limit = max(0.0, budget * GB - non_kv)
    kv_actual = min(kv_theory, kv_limit)
    # 按实际 KV 空间反推最大并发数(类似 vLLM 的 Available KV cache memory / per_seq)
    max_kv_seqs = int(kv_actual / kv_per_seq) if kv_per_seq > 0 else 0

    total = non_kv + kv_actual
    per_gpu = (weights + kv_actual) / n_gpu + act / n_gpu + overhead_per_gpu

    breakdown = MemoryBreakdown(
        weights_gb=weights / GB,
        kv_cache_gb=kv_actual / GB,
        kv_cache_limit_gb=kv_limit / GB,
        activations_gb=act / GB,
        overhead_gb=overhead / GB,
        total_gb=total / GB,
        per_gpu_gb=per_gpu / GB,
        max_kv_seqs=max_kv_seqs,
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

    # ---- 互联通信损耗 ----
    internode_overhead_mult = 1.0
    if n_gpu > GPUS_PER_NODE:
        # 多机/跨机部署: 损耗取决于跨机互联类型
        n_nodes = math.ceil(n_gpu / GPUS_PER_NODE)
        f = INTERNODE_COMPUTE_FACTOR.get(inf.internode, 0.90)
        compute_util *= f
        internode_overhead_mult = INTERNODE_OVERHEAD_MULT.get(inf.internode, 1.4)
        if f < 1.0:
            warnings.append(
                f"{n_nodes} 机跨机部署({inf.internode}): 跨机通信已下调算力效率 "
                f"{int(round((1 - f) * 100))}%。若为 NVLink Switch/HCCS 等高速无损互联,"
                "可将'跨机互联'选为'高速无损'消除此损耗。"
            )
        else:
            warnings.append(
                f"{n_nodes} 机跨机部署: 已按高速无损互联(NVLink Switch/HCCS)计,跨机通信几乎无损耗。"
            )
    elif n_gpu > 1 and not all_nvlink:
        # 单机但无机内高速互联(PCIe 等)
        compute_util *= 0.85
        warnings.append("机内无高速互联(NVLink/HCCS),TP 通信开销已下调效率 ~15%。")

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
        * internode_overhead_mult
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
    max_fit_seqs = mem.max_kv_seqs

    # ---------------- 瓶颈分类 + 按实际参数生成优化建议 ----------------
    bottleneck = _classify_bottleneck(mem, total_mem_gb, tpot, decode_compute_t)
    suggestions = _build_suggestions(
        model=model,
        inf=inf,
        mem=mem,
        n_gpu=n_gpu,
        batch=batch,
        util=util,
        fits=fits,
        max_fit_seqs=max_fit_seqs,
        bottleneck=bottleneck,
        tpot_bw=tpot,
        decode_compute_t=decode_compute_t,
        tpot_roof=tpot_roof,
        overhead_s=overhead_s,
        single_tps=single_tps,
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


def _classify_bottleneck(
    mem: MemoryBreakdown, total_mem_gb: float, tpot_bw: float, decode_compute_t: float
) -> str:
    util = mem.total_gb / total_mem_gb if total_mem_gb > 0 else 0.0
    if util > 0.95:
        return "Memory Bound"
    return "Bandwidth Bound" if tpot_bw >= decode_compute_t else "Compute Bound"


def _build_suggestions(
    *,
    model: ModelSpec,
    inf: InferenceConfig,
    mem: MemoryBreakdown,
    n_gpu: int,
    batch: int,
    util: float,
    fits: bool,
    max_fit_seqs: int,
    bottleneck: str,
    tpot_bw: float,
    decode_compute_t: float,
    tpot_roof: float,
    overhead_s: float,
    single_tps: float,
) -> list[str]:
    """根据当前实际参数生成可操作的优化建议(非通用模板)。"""
    tips: list[str] = []
    w = mem.weights_gb

    # 1) 放不下: 最高优先, 解决后其它才有意义
    if not fits:
        fix = [f"增加卡数(当前 {n_gpu})"]
        if inf.quantization == "none":
            fix.append(f"启用 fp8 量化(权重 {w:.0f}→约 {w/2:.0f}GB)")
        fix.append("或换更大显存卡")
        tips.append("❗ 放不下:权重+激活已占满单卡预算。先解决显存——" + "、".join(fix) + "。")
        return tips

    # 2) 并发 vs 显存
    if max_fit_seqs < batch:
        opts = []
        if inf.kv_cache_dtype == "auto":
            opts.append("--kv-cache-dtype 设 fp8(KV 减半→可容纳翻倍)")
        if inf.gpu_memory_utilization < 0.9:
            opts.append(f"--gpu-memory-utilization 由 {inf.gpu_memory_utilization} 提到 0.9")
        opts.append(f"或把 --max-num-seqs 由 {batch} 降到 {max_fit_seqs}(免排队)")
        tips.append(
            f"⚠ max-num-seqs={batch} 实际仅能容纳约 {max_fit_seqs} 路(KV 空间不足),多余请求会排队。可:"
            + "；".join(opts) + "。"
        )
    elif util < 0.55 and max_fit_seqs > batch:
        tips.append(
            f"显存仅用 {util*100:.0f}%,还能容纳约 {max_fit_seqs} 路。把 --max-num-seqs 由 {batch} "
            f"提到约 {max_fit_seqs} 可大幅提升总吞吐(单请求速度不变)。"
        )

    # 3) enforce-eager(区分架构)
    if inf.enforce_eager:
        if model.is_moe or model.is_linear_attn:
            tips.append(
                "该模型为 MoE/线性注意力架构,当前 vLLM 通常必须 --enforce-eager;"
                "单请求提速主要靠升级 vLLM 或等 kernel 优化,堆硬件收效有限。"
            )
        else:
            denom = tpot_roof + overhead_s * CUDA_GRAPH_FACTOR
            est = 1.0 / denom if denom > 0 else 0.0
            if est > single_tps * 1.2:
                tips.append(
                    f"模型是 dense,可关闭 --enforce-eager(启用 CUDA Graph):"
                    f"单请求 TPS 预计由约 {single_tps:.0f} 提升到约 {est:.0f} tok/s。"
                )

    # 4) 量化(未量化且权重占比可观时才提)
    if inf.quantization == "none" and w >= 2:
        tips.append(
            f"未量化。启用 fp8 量化可把权重 {w:.0f}GB 减半、把省下的显存给 KV(可容纳并发约翻倍),"
            "H 卡还能加速,精度损失通常很小。"
        )

    # 5) KV 精度(KV 是显存大头, 且并发没受限时)
    if inf.kv_cache_dtype == "auto" and mem.kv_cache_gb > max(w, 1) and max_fit_seqs >= batch:
        tips.append(
            f"KV Cache({mem.kv_cache_gb:.0f}GB)是显存大头。设 --kv-cache-dtype fp8 可让 KV 减半、"
            "可容纳并发约翻倍。"
        )

    # 6) 瓶颈方向
    if bottleneck == "Bandwidth Bound":
        ratio = tpot_bw / max(decode_compute_t, 1e-9)
        tips.append(
            f"瓶颈在显存带宽(约 {ratio:.1f}× 于算力):提高 max-num-seqs 摊薄权重读取可提升总吞吐;"
            "单请求要更快需更高带宽卡(H20/H200)。"
        )
    elif bottleneck == "Compute Bound":
        ratio = decode_compute_t / max(tpot_bw, 1e-9)
        tips.append(
            f"瓶颈在算力(约 {ratio:.1f}× 于带宽):增加卡数或换更高算力卡(H100/H200)。"
        )

    return tips[:5] if tips else ["当前配置较均衡,无明显瓶颈;可逐步提高 max-num-seqs 压测真实吞吐。"]
