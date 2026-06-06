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
  head_dim?: number | null;
  vocab_size: number;
  precision: string;
  tensor_types?: string;
  weight_size_gb?: number | null;
  params_accurate?: boolean;
  sliding_window?: number | null;
  num_full_attention_layers?: number;
  num_global_key_value_heads?: number | null;
  global_head_dim?: number | null;
  active_params_b?: number | null;
  is_moe?: boolean;
  is_linear_attn?: boolean;
  kv_cache_factor?: number;
  attn_type?: string;
  mla_kv_dim?: number;
}

export interface GpuGroup {
  spec: GpuSpec;
  count: number;
}

export type DType = "auto" | "float16" | "bfloat16" | "float32";
export type Quantization =
  | "none"
  | "fp8"
  | "awq"
  | "gptq"
  | "int8"
  | "w8a8"
  | "w4a8"
  | "w4a16";
export type KVCacheDType = "auto" | "fp8" | "fp8_e5m2" | "fp8_e4m3" | "int8";
export type InterNode = "nvlink" | "ib" | "ethernet";
export type IntraNode = "auto" | "pcie";

export interface InferenceConfig {
  max_model_len: number;
  max_num_seqs: number;
  max_num_batched_tokens: number;
  input_len: number;
  dtype: DType;
  quantization: Quantization;
  kv_cache_dtype: KVCacheDType;
  gpu_memory_utilization: number;
  enforce_eager: boolean;
  intra_node: IntraNode;
  gpus_per_node: number;
  internode: InterNode;
  async_scheduling: boolean;
  parallel_enabled: boolean;
  tp_size: number;
  pp_size: number;
  dp_size: number;
  mem_util?: number | null;
  compute_util?: number | null;
}

export interface MemoryBreakdown {
  weights_gb: number;
  kv_cache_gb: number;
  kv_cache_limit_gb: number;
  activations_gb: number;
  overhead_gb: number;
  total_gb: number;
  per_gpu_gb: number;
  max_kv_seqs: number;
}

export interface EstimateResponse {
  memory: MemoryBreakdown;
  total_mem_gb: number;
  mem_utilization: number;
  fits: boolean;
  tps: number;
  tps_low: number;
  tps_high: number;
  single_tps: number;
  single_tps_low: number;
  single_tps_high: number;
  ttft_ms: number;
  tpot_ms: number;
  request_latency_ms: number;
  max_fit_seqs: number;
  bottleneck: "Compute Bound" | "Memory Bound" | "Bandwidth Bound";
  suggestions: string[];
  analysis_reliable: boolean;
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
  weight_size_gb?: number | null;
  error?: string;
}
