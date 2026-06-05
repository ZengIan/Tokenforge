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


def _parallel_sizes(inf: InferenceConfig, n_gpu: int) -> tuple[int, int, int]:
    """返回 (tp, pp, dp)。未启用并行配置时 TP=总卡数, PP=DP=1。"""
    if inf.parallel_enabled:
        return max(1, inf.tp_size), max(1, inf.pp_size), max(1, inf.dp_size)
    return max(1, n_gpu), 1, 1


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

    if model.attn_type == "MLA" and model.mla_kv_dim > 0:
        # MLA(DeepSeek/GLM-MLA): 每层每 token 只缓存 1 个低秩 latent(kv_lora_rank+rope),
        # 与注意力头数解耦, 不 ×2(V 由 latent 还原), 比 MHA 小 10~100 倍。
        kv_per_seq = (
            model.num_layers * model.mla_kv_dim * seq * kv_b / KV_UTIL
        )
        kv_theory = kv_per_seq * active_seqs
    elif model.sliding_window and model.num_full_attention_layers > 0:
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

    # 并行切分: 权重/KV 按模型并行(TP×PP)分片; DP 复制整套权重
    tp, pp, dp = _parallel_sizes(inf, n_gpu)
    mp = max(1, tp * pp)
    weights_cluster = weights * dp  # DP 复制 dp 份权重

    non_kv = weights_cluster + act + overhead
    # 可用于 KV 的预算空间, 用于反推"实际可容纳并发"(vLLM 会按此截断并发)
    kv_limit = max(0.0, budget * GB - non_kv)
    max_kv_seqs = int(kv_limit / kv_per_seq) if kv_per_seq > 0 else 0

    # 显示口径用"真实需求"(KV 按 max_model_len×max_num_seqs 满算, 不截断),
    # 这样"总显存占用/利用率"才有意义(可 >100%, 表示放不下→会降并发到 max_kv_seqs)。
    total = non_kv + kv_theory
    # 单卡: 权重按模型并行 mp 分片(DP 不减单卡权重); KV/激活 摊到所有卡
    per_gpu = weights / mp + (kv_theory + act) / n_gpu + overhead_per_gpu

    breakdown = MemoryBreakdown(
        weights_gb=weights_cluster / GB,
        kv_cache_gb=kv_theory / GB,
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

    # ---- 并行配置 + 合理性校验 ----
    tp, pp, dp = _parallel_sizes(inf, n_gpu)
    mp = max(1, tp * pp)
    if inf.parallel_enabled:
        world = tp * pp * dp
        if world != n_gpu:
            warnings.append(
                f"⚠ TP×PP×DP = {tp}×{pp}×{dp} = {world} 与总卡数 {n_gpu} 不一致"
                "(vLLM 要求三者乘积 = 总卡数),模型将无法启动。"
            )
        if model.num_attention_heads % tp != 0:
            warnings.append(
                f"⚠ 注意力头数 {model.num_attention_heads} 不能被 TP={tp} 整除,"
                "张量并行无法切分,模型无法启动(需 num_attention_heads % TP == 0)。"
            )
        kvh = _kv_heads(model)
        if kvh % tp != 0 and tp % kvh != 0:
            warnings.append(
                f"⚠ KV 头数 {kvh} 与 TP={tp} 不整除,GQA 切分可能失败。"
            )
        if model.num_layers % pp != 0:
            warnings.append(
                f"⚠ 层数 {model.num_layers} 不能被 PP={pp} 整除,流水线切分不均,可能无法启动。"
            )
        if inf.max_num_batched_tokens < pp:
            warnings.append(
                f"⚠ max-num-batched-tokens({inf.max_num_batched_tokens}) 小于 PP={pp},"
                "流水线每个 micro-batch 不足 1 token,无法启动。"
            )

    # ---- 互联通信损耗 ----
    internode_overhead_mult = 1.0
    if n_gpu > inf.gpus_per_node:
        # 多机/跨机: 损耗取决于跨机互联类型
        n_nodes = math.ceil(n_gpu / inf.gpus_per_node)
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
    elif n_gpu > 1:
        # 单机多卡: 机内互联 (auto 按卡库 nvlink, 否则按用户选择)
        has_highspeed = (
            all_nvlink if inf.intra_node == "auto" else inf.intra_node == "highspeed"
        )
        if not has_highspeed:
            compute_util *= 0.85
            warnings.append(
                "机内按纯 PCIe(无 NVLink/HCCS)计,TP 通信开销已下调效率 ~15%;"
                "若卡间有 HCCS/NVLink 链路,请把'机内互联'选为'高速'。"
            )

    batch = inf.max_num_seqs  # continuous batching: in-flight sequences
    # 实际同时在跑的路数: 受 KV 显存限制(放不下的请求会排队, 不计入瞬时吞吐)
    eff_batch = min(batch, max(1, mem.max_kv_seqs))

    # MoE: decode/prefill 只用激活参数(全部专家常驻显存, 但每 token 只读 top-k)
    active_params = (model.active_params_b or model.params_b) * 1e9
    active_ratio = active_params / params if params > 0 else 1.0

    # 单请求只用其所在副本的 模型并行(mp=TP×PP) 组算力/带宽; DP 副本并行服务不同请求
    mp_frac = mp / n_gpu  # 一个副本占总卡数的比例
    bw_replica = bw_total * mp_frac    # 单副本聚合带宽
    flops_replica = flops_total * mp_frac

    # ---------------- DECODE 屋顶线 ----------------
    # 每步搬运: 激活权重(整批共享一次) + 每条序列各自的 KV(随上下文长度增长)。
    weight_read_bytes = mem.weights_gb * GB * active_ratio / max(1, dp)  # 单副本权重
    # 每路 KV 读取量按"实际序列长度"(≈输入长度)缩放; mem.kv_cache_gb 是 max-model-len 满算口径
    in_len = min(inf.input_len, inf.max_model_len)
    kv_per_seq = (mem.kv_cache_gb * GB / max(1, batch)) * (in_len / max(1, inf.max_model_len))
    # 单请求: 权重 + 自己 1 路 KV; 总吞吐: 权重(共享) + 实际在跑的 eff_batch 路 KV
    tpot_bw_single = (weight_read_bytes + kv_per_seq) / (bw_replica * mem_util)
    tpot_bw_total = (weight_read_bytes + kv_per_seq * eff_batch) / (bw_total * mem_util)
    decode_compute_t = (2 * active_params * eff_batch) / (flops_total * compute_util)

    # ---------------- 每 token 固定开销(现实地板) ----------------
    arch_mult = 1.0
    if model.is_moe:
        arch_mult *= MOE_OVERHEAD_MULT
    if model.is_linear_attn:
        arch_mult *= LINEAR_ATTN_OVERHEAD_MULT
    # TP all-reduce(随 TP 规模)+ PP bubble
    tp_mult = 1.0 + 0.15 * (tp - 1) + 0.05 * (pp - 1)
    graph_mult = 1.0 if inf.enforce_eager else CUDA_GRAPH_FACTOR
    overhead_s = (
        model.num_layers * PER_LAYER_OVERHEAD_MS * arch_mult * tp_mult * graph_mult
        * internode_overhead_mult
    ) / 1000.0

    # 单请求 TPOT(副本带宽) 与 总吞吐 TPOT(聚合带宽)
    tpot_single = max(tpot_bw_single, decode_compute_t / max(1, eff_batch)) + overhead_s
    tpot_total = max(tpot_bw_total, decode_compute_t) + overhead_s

    # ---------------- PREFILL → TTFT ----------------
    # TTFT 是算力瓶颈, 同时随 ① 输入(prompt)长度 ② 并发(prefill 争抢算力) 增长。
    in_len = min(inf.input_len, inf.max_model_len)  # prompt 不超过上下文
    #   FFN/投影(线性): 2 × 激活参数 × prompt; MoE 按激活参数
    ffn_flops = 2 * active_params * in_len
    #   注意力(QK^T + AV): ~4 × prompt² × hidden × 层数, 长 prompt 时主导
    attn_flops = 4 * (in_len ** 2) * model.hidden_size * model.num_layers
    prefill_flops = ffn_flops + attn_flops
    # 单请求(低负载)prefill 计算时间
    ttft_single = prefill_flops / (flops_replica * compute_util) * (1.0 + 0.10 * (pp - 1))
    # 并发争抢: prefill 算力守恒, B 路并发时按 FIFO 平均排队 (B+1)/2 拉高 TTFT
    prefill_contention = (eff_batch + 1) / 2.0
    ttft = ttft_single * prefill_contention + overhead_s

    def _safe(x: float) -> float:
        return 1.0 / x if x > 0 else 0.0

    # 区间: 固定开销不确定性 (单请求口径)
    single_tps = _safe(tpot_single)
    single_tps_high = _safe(tpot_bw_single + overhead_s * OVERHEAD_FAST)
    single_tps_low = _safe(tpot_bw_single + overhead_s * OVERHEAD_SLOW)
    tps = eff_batch * _safe(tpot_total)
    tps_high = eff_batch * _safe(tpot_bw_total + overhead_s * OVERHEAD_FAST)
    tps_low = eff_batch * _safe(tpot_bw_total + overhead_s * OVERHEAD_SLOW)
    tpot_eff = tpot_single
    request_latency = ttft + tpot_single  # 首字 + 单 token
    # 供瓶颈分析判断带宽 vs 算力
    tpot = tpot_bw_total

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
        tpot_roof=tpot_bw_single,
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

    # 1) 连 1 路都放不下: 最高优先(权重+激活+单路KV 就超预算)
    if max_fit_seqs < 1:
        fix = [f"增加卡数(当前 {n_gpu})"]
        if inf.quantization == "none":
            fix.append(f"启用 fp8 量化(权重 {w:.0f}→约 {w/2:.0f}GB)")
        if inf.kv_cache_dtype == "auto":
            fix.append("--kv-cache-dtype 设 fp8")
        fix.append("或减小 max-model-len / 换更大显存卡")
        tips.append("❗ 放不下:权重+激活+单路 KV 已超单卡预算。先解决显存——" + "、".join(fix) + "。")
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
            f"⚠ 显存放不下 {batch} 路并发,实际仅能跑约 {max_fit_seqs} 路(vLLM 会自动降并发/排队)。可:"
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
