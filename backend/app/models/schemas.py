"""Pydantic request/response schemas for the Tokenforge API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Quant = Literal[
    "FP16", "BF16", "FP8", "INT8", "INT4", "GPTQ", "AWQ", "W8A8", "W4A8", "W4A16"
]
KVQuant = Literal["FP16", "BF16", "FP8", "INT8"]
Framework = Literal["vLLM", "TensorRT-LLM", "SGLang", "llama.cpp"]


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
    vocab_size: int = 32000
    # weight storage precision of the checkpoint, used as a fallback for quant
    precision: Quant = "FP16"


class GpuGroup(BaseModel):
    """One row of the GPU config table: a card type + how many."""

    spec: GpuSpec
    count: int = Field(1, ge=1, le=1024)


class InferenceConfig(BaseModel):
    input_len: int = Field(1024, ge=1)
    output_len: int = Field(256, ge=1)
    # freely-configurable context window used to size the KV cache (8k–256k)
    context_len: int = Field(8192, ge=8192, le=262144)
    concurrency: int = Field(1, ge=1, le=4096)
    batch_size: int = Field(1, ge=1, le=1024)
    quant: Quant = "FP16"
    kv_quant: KVQuant = "FP16"
    framework: Framework = "vLLM"
    # vLLM-style serving knobs
    gpu_memory_utilization: float = Field(0.90, ge=0.1, le=1.0)
    enforce_eager: bool = False
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
    activations_gb: float
    overhead_gb: float
    total_gb: float
    per_gpu_gb: float


class EstimateResponse(BaseModel):
    # memory
    memory: MemoryBreakdown
    total_mem_gb: float
    mem_utilization: float
    fits: bool
    # throughput
    tps: float
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    # analysis
    bottleneck: Literal["Compute Bound", "Memory Bound", "Bandwidth Bound"]
    suggestions: list[str]
    # echo of effective coefficients used
    effective_compute_util: float
    effective_mem_util: float
    warnings: list[str] = []


class CompareRequest(BaseModel):
    scenarios: list[EstimateRequest]
