import type {
  EstimateResponse,
  GpuGroup,
  GpuSpec,
  InferenceConfig,
  ModelSpec,
  SearchResult,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE || "";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export async function fetchGpus(): Promise<GpuSpec[]> {
  return json(await fetch(`${BASE}/api/gpus`));
}

export async function searchModels(q: string): Promise<SearchResult[]> {
  const res = await json<{ results: SearchResult[] }>(
    await fetch(`${BASE}/api/models/search?q=${encodeURIComponent(q)}`)
  );
  return res.results;
}

export async function fetchModelDetail(
  modelId: string
): Promise<ModelSpec & { config_found: boolean }> {
  return json(await fetch(`${BASE}/api/models/${modelId}`));
}

export async function estimate(payload: {
  model: ModelSpec;
  gpus: GpuGroup[];
  inference: InferenceConfig;
}): Promise<EstimateResponse> {
  return json(
    await fetch(`${BASE}/api/estimate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}
