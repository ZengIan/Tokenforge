import { create } from "zustand";
import type {
  EstimateResponse,
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
  max_num_batched_tokens: 8192,
  dtype: "auto",
  quantization: "none",
  kv_cache_dtype: "auto",
  gpu_memory_utilization: 0.9,
  enforce_eager: false,
};

interface State {
  gpuDb: GpuSpec[];
  model: ModelSpec;
  gpuGroups: GpuGroup[];
  inference: InferenceConfig;
  result: EstimateResponse | null;
  loading: boolean;
  error: string | null;

  setGpuDb: (g: GpuSpec[]) => void;
  setModel: (m: Partial<ModelSpec>) => void;
  setInference: (i: Partial<InferenceConfig>) => void;
  addGpuGroup: (spec: GpuSpec) => void;
  updateGpuGroup: (idx: number, patch: Partial<GpuGroup>) => void;
  removeGpuGroup: (idx: number) => void;
  setResult: (r: EstimateResponse | null) => void;
  setLoading: (b: boolean) => void;
  setError: (e: string | null) => void;
  hydrateFromUrl: () => void;
  serializeToUrl: () => string;
}

export const useStore = create<State>((set, get) => ({
  gpuDb: [],
  model: DEFAULT_MODEL,
  gpuGroups: [],
  inference: DEFAULT_INFERENCE,
  result: null,
  loading: false,
  error: null,

  setGpuDb: (g) =>
    set((s) => ({
      gpuDb: g,
      // seed one group with the first card if empty
      gpuGroups:
        s.gpuGroups.length > 0 || g.length === 0
          ? s.gpuGroups
          : [{ spec: g[0], count: 8 }],
    })),
  setModel: (m) => set((s) => ({ model: { ...s.model, ...m } })),
  setInference: (i) => set((s) => ({ inference: { ...s.inference, ...i } })),
  addGpuGroup: (spec) =>
    set((s) => ({ gpuGroups: [...s.gpuGroups, { spec, count: 1 }] })),
  updateGpuGroup: (idx, patch) =>
    set((s) => ({
      gpuGroups: s.gpuGroups.map((g, i) => (i === idx ? { ...g, ...patch } : g)),
    })),
  removeGpuGroup: (idx) =>
    set((s) => ({ gpuGroups: s.gpuGroups.filter((_, i) => i !== idx) })),
  setResult: (r) => set({ result: r }),
  setLoading: (b) => set({ loading: b }),
  setError: (e) => set({ error: e }),

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
      set((s) => ({
        model: { ...s.model, ...payload.m },
        inference: { ...s.inference, ...payload.i },
        gpuGroups: groups.length ? groups : s.gpuGroups,
      }));
    } catch {
      /* ignore malformed share links */
    }
  },
}));
