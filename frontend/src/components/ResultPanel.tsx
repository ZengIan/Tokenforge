import { useStore } from "../store";
import type { EstimateResponse } from "../types";

export function ResultPanel() {
  const { result, loading, error } = useStore();

  if (error) {
    return <div className="card text-amber-400">估算失败：{error}</div>;
  }
  if (!result) {
    return (
      <div className="card flex h-40 items-center justify-center text-slate-500">
        {loading ? "锻造中… 🔥" : "调整左侧参数即可自动估算"}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <MetricCards r={result} />
      <MemoryBreakdownChart r={result} />
      <Bottleneck r={result} />
      {result.warnings.length > 0 && (
        <div className="card border-amber-700/50 bg-amber-950/20">
          <h3 className="mb-1 text-xs font-bold text-amber-400">提示</h3>
          <ul className="list-disc space-y-0.5 pl-4 text-xs text-amber-200/80">
            {result.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function MetricCards({ r }: { r: EstimateResponse }) {
  const cards = [
    { label: "TPS", value: fmt(r.tps), unit: "tokens/s", accent: true },
    { label: "总显存占用", value: r.memory.total_gb.toFixed(1), unit: "GB" },
    { label: "单卡占用", value: r.memory.per_gpu_gb.toFixed(1), unit: "GB" },
    { label: "显存利用率", value: (r.mem_utilization * 100).toFixed(0), unit: "%" },
    { label: "TTFT 首字延迟", value: r.ttft_ms.toFixed(0), unit: "ms" },
    { label: "TPOT 每字延迟", value: r.tpot_ms.toFixed(1), unit: "ms" },
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {cards.map((c) => (
        <div
          key={c.label}
          className={
            "card p-3 " + (c.accent ? "border-forge-ember/60 bg-forge-ember/10" : "")
          }
        >
          <div className="text-[11px] text-slate-400">{c.label}</div>
          <div className={"mt-1 text-xl font-bold " + (c.accent ? "text-forge-ember" : "text-slate-100")}>
            {c.value}
            <span className="ml-1 text-xs font-normal text-slate-500">{c.unit}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

const SEGMENTS = [
  { key: "weights_gb", label: "模型权重", color: "bg-forge-ember" },
  { key: "kv_cache_gb", label: "KV Cache", color: "bg-sky-500" },
  { key: "activations_gb", label: "激活值", color: "bg-violet-500" },
  { key: "overhead_gb", label: "框架开销", color: "bg-slate-500" },
] as const;

function MemoryBreakdownChart({ r }: { r: EstimateResponse }) {
  const total = r.memory.total_gb || 1;
  return (
    <div className="card">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-bold text-slate-300">显存分解</h3>
        <span className={"text-xs " + (r.fits ? "text-emerald-400" : "text-red-400")}>
          {r.fits ? "✓ 可容纳" : "✗ 单卡超限"} · 容量 {r.total_mem_gb.toFixed(0)}GB
        </span>
      </div>
      <div className="flex h-6 w-full overflow-hidden rounded-lg bg-slate-900">
        {SEGMENTS.map((s) => {
          const v = r.memory[s.key];
          const pct = (v / total) * 100;
          return pct > 0.5 ? (
            <div
              key={s.key}
              className={s.color}
              style={{ width: `${pct}%` }}
              title={`${s.label}: ${v.toFixed(1)}GB`}
            />
          ) : null;
        })}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-1 text-[11px] sm:grid-cols-4">
        {SEGMENTS.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5">
            <span className={"inline-block h-2.5 w-2.5 rounded-sm " + s.color} />
            <span className="text-slate-400">{s.label}</span>
            <span className="text-slate-200">{r.memory[s.key].toFixed(1)}GB</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Bottleneck({ r }: { r: EstimateResponse }) {
  const color =
    r.bottleneck === "Memory Bound"
      ? "text-red-400 border-red-700/50"
      : r.bottleneck === "Bandwidth Bound"
      ? "text-sky-400 border-sky-700/50"
      : "text-amber-400 border-amber-700/50";
  return (
    <div className={"card " + color}>
      <h3 className="text-xs font-bold">瓶颈分析 · {r.bottleneck}</h3>
      <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-xs text-slate-300">
        {r.suggestions.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
      <p className="mt-2 text-[11px] text-slate-500">
        计算系数：算力利用率 {(r.effective_compute_util * 100).toFixed(0)}% · 带宽利用率{" "}
        {(r.effective_mem_util * 100).toFixed(0)}%
      </p>
    </div>
  );
}

function fmt(n: number) {
  return n >= 1000 ? Math.round(n).toLocaleString() : n.toFixed(1);
}
