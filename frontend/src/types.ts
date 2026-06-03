export interface GpuSpec {
  name: string;
  mem_gb: number;
  bw_gbs: number;
  fp16_tflops: number;
  fp8_tflops: number;
  nvlink: boolean;
  source: string;
  note: string;
}

export interface ModelSpec {
  model_id: string;
  params_b: number;
  hidden_size: number;
  num_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number | null;
  vocab_size: number;
  precision: string;
}

export interface GpuGroup {
  spec: GpuSpec;
  count: number;
}

export type Quant = "FP16" | "BF16" | "FP8" | "INT8" | "INT4" | "GPTQ" | "AWQ";
export type Framework = "vLLM" | "TensorRT-LLM" | "SGLang" | "llama.cpp";

export interface InferenceConfig {
  input_len: number;
  output_len: number;
  concurrency: number;
  batch_size: number;
  quant: Quant;
  kv_quant: "FP16" | "BF16" | "FP8" | "INT8";
  framework: Framework;
  mem_util?: number | null;
  compute_util?: number | null;
}

export interface MemoryBreakdown {
  weights_gb: number;
  kv_cache_gb: number;
  activations_gb: number;
  overhead_gb: number;
  total_gb: number;
  per_gpu_gb: number;
}

export interface EstimateResponse {
  memory: MemoryBreakdown;
  total_mem_gb: number;
  mem_utilization: number;
  fits: boolean;
  tps: number;
  ttft_ms: number;
  tpot_ms: number;
  request_latency_ms: number;
  bottleneck: "Compute Bound" | "Memory Bound" | "Bandwidth Bound";
  suggestions: string[];
  effective_compute_util: number;
  effective_mem_util: number;
  warnings: string[];
}

export interface SearchResult {
  model_id: string;
  name: string;
  chinese_name?: string;
  params_b: number | null;
  precision: string;
  task: string;
  downloads: number;
  error?: string;
}
