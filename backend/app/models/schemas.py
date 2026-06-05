"""Pydantic request/response schemas for the Tokenforge API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --dtype: 计算精度
DType = Literal["auto", "float16", "bfloat16", "float32"]
# --quantization: 权重量化方法 (vLLM 取值风格)
Quantization = Literal[
    "none", "fp8", "awq", "gptq", "int8", "w8a8", "w4a8", "w4a16"
]
# --kv-cache-dtype: KV 缓存精度
KVCacheDType = Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3", "int8"]
# 跨机(多机)互联类型: 卡数 > 单机上限时生效
InterNode = Literal["nvlink", "ib", "ethernet"]


class GpuSpec(BaseModel):
    """A single GPU model's hardware spec (mirrors gpus.yaml)."""

    name: str
    mem_gb: float
    bw_gbs: float
    fp16_tflops: float
    fp8_tflops: float = 0
    nvlink: bool = False
    source: str = "estimate"
    note: str = ""


class ModelSpec(BaseModel):
    """Architecture parameters needed to estimate memory / throughput."""

    model_id: str = ""
    params_b: float = Field(..., description="Parameter count in billions, e.g. 7 for 7B")
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # GQA; defaults to num_attention_heads
    head_dim: Optional[int] = None  # 若 config.json 直接提供则优先使用
    vocab_size: int = 32000
    # 默认模型精度 (从 config.json torch_dtype 读取)
    precision: str = "FP16"
    # 张量类型 (从 config.json quantization_config 提取的完整精度信息)
    tensor_types: str = ""
    # 真实权重文件大小 (GB),来自 ModelScope 仓库文件汇总;为空则前端按公式估算
    weight_size_gb: Optional[float] = None
    # 参数量是否从 ModelScope API 准确获取
    params_accurate: bool = False
    # --- 混合注意力 (如 Gemma4) ---
    sliding_window: Optional[int] = None  # sliding attention 窗口大小
    num_full_attention_layers: int = 0  # full_attention 层数;0 表示全是 full/sliding 统一处理
    num_global_key_value_heads: Optional[int] = None  # full_attention 层的 KV head 数
    global_head_dim: Optional[int] = None  # full_attention 层的 head_dim
    # --- 架构特征(影响吞吐建模) ---
    # MoE 每 token 激活参数量(B);None 表示 dense(=params_b)。decode 只读激活权重。
    active_params_b: Optional[float] = None
    is_moe: bool = False
    # 线性/混合注意力(如 Qwen3-Next 的 Gated DeltaNet):KV cache 远小于普通注意力
    is_linear_attn: bool = False
    # KV cache 系数(相对普通全注意力):1.0=全注意力, ~0.25=混合, ~0.05=纯线性
    kv_cache_factor: float = 1.0


class GpuGroup(BaseModel):
    """One row of the GPU config table: a card type + how many."""

    spec: GpuSpec
    count: int = Field(1, ge=1, le=1024)


class InferenceConfig(BaseModel):
    """对应 vLLM 启动参数。默认值即官方推荐配置。"""

    # --max-model-len: 单请求最大上下文长度,决定 KV Cache 每路预留
    max_model_len: int = Field(8192, ge=512, le=1048576)
    # --max-num-seqs: 引擎最大并发序列数
    max_num_seqs: int = Field(256, ge=1, le=8192)
    # --max-num-batched-tokens: 单次迭代最多处理 token 数 (prefill 分块)
    max_num_batched_tokens: int = Field(8192, ge=256, le=1048576)
    # 输入长度(Prompt tokens): 仅用于估算 TTFT(首字延迟随它线性增长)
    input_len: int = Field(2048, ge=1, le=1048576)
    # --dtype: 计算精度
    dtype: DType = "auto"
    # --quantization: 权重量化方法
    quantization: Quantization = "none"
    # --kv-cache-dtype: KV 缓存精度
    kv_cache_dtype: KVCacheDType = "auto"
    # --gpu-memory-utilization: 单卡显存使用上限占比
    gpu_memory_utilization: float = Field(0.90, ge=0.1, le=1.0)
    # --enforce-eager: 关闭 CUDA Graph
    enforce_eager: bool = False
    # 机内互联: auto=按卡库 nvlink 字段; highspeed=有 NVLink/HCCS(无损); pcie=纯 PCIe(-15%)
    intra_node: Literal["auto", "highspeed", "pcie"] = "auto"
    # 单机卡数(超过即视为多机/跨机), 如 PPU 单机 16 卡可设 16
    gpus_per_node: int = Field(8, ge=1, le=64)
    # 跨机互联(仅卡数 > gpus_per_node 的多机部署生效):
    #   nvlink = NVLink Switch/HCCS 等高速无损互联(几乎无损)
    #   ib = InfiniBand/RoCE 高速网; ethernet = 普通以太网
    internode: InterNode = "ib"
    # --- 并行配置(可选; 不启用时 TP=总卡数, PP=DP=1) ---
    parallel_enabled: bool = False
    tp_size: int = Field(1, ge=1, le=1024)  # --tensor-parallel-size
    pp_size: int = Field(1, ge=1, le=256)   # --pipeline-parallel-size
    dp_size: int = Field(1, ge=1, le=256)   # --data-parallel-size
    # optional manual override of bandwidth/compute utilisation [0,1]
    mem_util: Optional[float] = None
    compute_util: Optional[float] = None


class EstimateRequest(BaseModel):
    model: ModelSpec
    gpus: list[GpuGroup]
    inference: InferenceConfig


class MemoryBreakdown(BaseModel):
    weights_gb: float
    kv_cache_gb: float
    kv_cache_limit_gb: float  # 显存预算下 KV cache 实际可用上限
    activations_gb: float
    overhead_gb: float
    total_gb: float
    per_gpu_gb: float
    max_kv_seqs: int  # 按实际 KV 预算可容纳的最大并发数


class EstimateResponse(BaseModel):
    # memory
    memory: MemoryBreakdown
    total_mem_gb: float
    mem_utilization: float
    fits: bool
    # throughput (mid = 区间中值; low/high = 保守区间)
    tps: float
    tps_low: float
    tps_high: float
    single_tps: float  # 单请求生成速度 (tokens/s)
    single_tps_low: float
    single_tps_high: float
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    # 显存预算实际可容纳的并发序列数 (vLLM 会自动收敛到此值附近)
    max_fit_seqs: int
    # analysis
    bottleneck: Literal["Compute Bound", "Memory Bound", "Bandwidth Bound"]
    suggestions: list[str]
    # 性能/瓶颈分析是否可靠 (取决于所选 GPU 算力/带宽规格是否真实)
    analysis_reliable: bool = True
    # echo of effective coefficients used
    effective_compute_util: float
    effective_mem_util: float
    warnings: list[str] = []


class CompareRequest(BaseModel):
    scenarios: list[EstimateRequest]
