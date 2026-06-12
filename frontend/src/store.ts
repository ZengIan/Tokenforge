import { create } from "zustand";
import { buildRecord, type ExportRecord } from "./lib/exportExcel";
import {
  cardHasNativeFp8,
  coerceQuant,
  engineForCard,
  mapLegacyQuant,
} from "./lib/engine";
import type {
  EstimateResponse,
  Engine,
  GpuGroup,
  GpuSpec,
  InferenceConfig,
  ModelSpec,
} from "./types";

export const DEFAULT_MODEL: ModelSpec = {
  model_id: "Qwen3-8B",
  params_b: 8.19,
  hidden_size: 4096,
  num_layers: 36,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  vocab_size: 152064,
  precision: "BF16",
  tensor_types: "BF16",
  weight_size_gb: 16.40,
};

// 默认值即 vLLM 官方推荐配置
export const DEFAULT_INFERENCE: InferenceConfig = {
  max_model_len: 8192,
  max_num_seqs: 16,
  max_num_batched_tokens: 2048,
  input_len: 2048,
  dtype: "auto",
  quantization: "none",
  kv_cache_dtype: "auto",
  gpu_memory_utilization: 0.9,
  enforce_eager: false,
  intra_node: "auto",
  gpus_per_node: 8,
  internode: "roce",
  parallel_enabled: false,
  tp_size: 1,
  pp_size: 1,
  dp_size: 1,
  async_scheduling: false,
  // 推理引擎 + SGLang/vLLM 专用 (随所选 GPU 自动切换, 见 updateGpuGroup)
  engine: "vllm",
  enable_dp_attention: false,
  enable_dp_lm_head: false,
  moe_a2a_backend: "none",
  deepep_mode: "auto",
  moe_dense_tp_size: 1,
  enable_expert_parallel: false,
  dist_init_addr: "",
};

interface State {
  gpuDb: GpuSpec[];
  model: ModelSpec;
  gpuGroups: GpuGroup[];
  inference: InferenceConfig;
  // 用户是否手动改过引擎; true 后换卡不再自动覆盖引擎
  engineTouched: boolean;
  result: EstimateResponse | null;
  loading: boolean;
  error: string | null;
  records: ExportRecord[];

  setGpuDb: (g: GpuSpec[]) => void;
  setModel: (m: Partial<ModelSpec>) => void;
  setInference: (i: Partial<InferenceConfig>) => void;
  setEngine: (e: Engine) => void;
  addGpuGroup: (spec: GpuSpec) => void;
  updateGpuGroup: (idx: number, patch: Partial<GpuGroup>) => void;
  removeGpuGroup: (idx: number) => void;
  setResult: (r: EstimateResponse | null) => void;
  setLoading: (b: boolean) => void;
  setError: (e: string | null) => void;
  addRecord: () => void;
  removeRecord: (id: string) => void;
  clearRecords: () => void;
  hydrateFromUrl: () => void;
  serializeToUrl: () => string;
}

/** 换卡时自动推断引擎并把量化收敛到合法值 (用户手动改过引擎则不动引擎) */
function autoEnginePatch(
  spec: GpuSpec,
  inf: InferenceConfig,
  engineTouched: boolean,
): Partial<InferenceConfig> {
  const engine = engineTouched ? inf.engine : engineForCard(spec.name);
  const hasFp8 = cardHasNativeFp8(spec);
  return { engine, quantization: coerceQuant(engine, hasFp8, inf.quantization) };
}

export const useStore = create<State>((set, get) => ({
  gpuDb: [],
  model: DEFAULT_MODEL,
  gpuGroups: [],
  inference: DEFAULT_INFERENCE,
  engineTouched: false,
  result: null,
  loading: false,
  error: null,
  records: [],

  setGpuDb: (g) =>
    set((s) => {
      if (s.gpuGroups.length > 0 || g.length === 0) return { gpuDb: g };
      // seed one group with the first card + 按该卡自动选引擎
      return {
        gpuDb: g,
        gpuGroups: [{ spec: g[0], count: 8 }],
        inference: { ...s.inference, ...autoEnginePatch(g[0], s.inference, s.engineTouched) },
      };
    }),
  setModel: (m) => set((s) => ({ model: { ...s.model, ...m } })),
  setInference: (i) => set((s) => ({ inference: { ...s.inference, ...i } })),
  setEngine: (e) =>
    set((s) => {
      const spec = s.gpuGroups[0]?.spec;
      const hasFp8 = cardHasNativeFp8(spec);
      return {
        engineTouched: true,
        inference: {
          ...s.inference,
          engine: e,
          quantization: coerceQuant(e, hasFp8, s.inference.quantization),
          // 切到非 sglang 引擎时关闭 DP-Attention, 避免遗留非法状态
          ...(e === "sglang" ? {} : { enable_dp_attention: false }),
        },
      };
    }),
  addGpuGroup: (spec) =>
    set((s) => ({ gpuGroups: [...s.gpuGroups, { spec, count: 1 }] })),
  updateGpuGroup: (idx, patch) =>
    set((s) => {
      const gpuGroups = s.gpuGroups.map((g, i) =>
        i === idx ? { ...g, ...patch } : g,
      );
      // 换了第 0 组的卡型 → 自动选引擎 + 收敛量化
      const inference =
        idx === 0 && patch.spec
          ? { ...s.inference, ...autoEnginePatch(patch.spec, s.inference, s.engineTouched) }
          : s.inference;
      return { gpuGroups, inference };
    }),
  removeGpuGroup: (idx) =>
    set((s) => ({ gpuGroups: s.gpuGroups.filter((_, i) => i !== idx) })),
  setResult: (r) => set({ result: r }),
  setLoading: (b) => set({ loading: b }),
  setError: (e) => set({ error: e }),

  addRecord: () => {
    const { model, gpuGroups, inference, result } = get();
    if (!result || gpuGroups.length === 0) return;
    const rec = buildRecord(model, gpuGroups, inference, result);
    set((s) => ({ records: [...s.records, rec] }));
  },
  removeRecord: (id) =>
    set((s) => ({ records: s.records.filter((r) => r.id !== id) })),
  clearRecords: () => set({ records: [] }),

  serializeToUrl: () => {
    const { model, gpuGroups, inference } = get();
    const payload = {
      m: model,
      g: gpuGroups.map((g) => ({ n: g.spec.name, c: g.count })),
      i: inference,
    };
    const enc = btoa(encodeURIComponent(JSON.stringify(payload)));
    const url = `${location.origin}${location.pathname}?c=${enc}`;
    return url;
  },

  hydrateFromUrl: () => {
    const params = new URLSearchParams(location.search);
    const c = params.get("c");
    if (!c) return;
    try {
      const payload = JSON.parse(decodeURIComponent(atob(c)));
      const db = get().gpuDb;
      const groups: GpuGroup[] = (payload.g || [])
        .map((x: { n: string; c: number }) => {
          const spec = db.find((s) => s.name === x.n);
          return spec ? { spec, count: x.c } : null;
        })
        .filter(Boolean);
      set((s) => {
        const inf: InferenceConfig = { ...s.inference, ...payload.i };
        // 旧分享链接里的废弃量化值(int8/w8a8/w4a8/w4a16) → 合法值
        if (inf.quantization) inf.quantization = mapLegacyQuant(inf.quantization);
        return {
          model: { ...s.model, ...payload.m },
          inference: inf,
          // 分享链接显式带了 engine 即视为用户已选定
          engineTouched: s.engineTouched || payload.i?.engine != null,
          gpuGroups: groups.length ? groups : s.gpuGroups,
        };
      });
    } catch {
      /* ignore malformed share links */
    }
  },
}));
