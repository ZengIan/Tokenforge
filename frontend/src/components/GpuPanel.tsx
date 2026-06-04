import { useStore } from "../store";

export function GpuPanel() {
  const { gpuDb, gpuGroups, updateGpuGroup } = useStore();
  const group = gpuGroups[0];

  function setGpu(name: string) {
    const spec = gpuDb.find((x) => x.name === name);
    if (spec) updateGpuGroup(0, { spec });
  }

  if (!group) {
    return <div className="card text-slate-300">加载 GPU 卡库中…</div>;
  }

  return (
    <div className="card">
      <h2 className="mb-3 text-sm font-bold text-forge-flame">② GPU 配置</h2>

      <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-2">
        <div className="flex items-center gap-2">
          <select
            className="input flex-1"
            value={group.spec.name}
            onChange={(e) => setGpu(e.target.value)}
          >
            {gpuDb.map((x) => (
              <option key={x.name} value={x.name}>
                {x.name}
              </option>
            ))}
          </select>
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-300">×</span>
            <input
              type="number"
              min={1}
              className="input w-16"
              value={group.count}
              onChange={(e) =>
                updateGpuGroup(0, { count: Math.max(1, Number(e.target.value)) })
              }
            />
          </div>
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px] text-slate-300">
          <span className="chip">{group.spec.mem_gb}GB</span>
          <span className="chip">{group.spec.bw_gbs} GB/s</span>
          <span className="chip">FP16 {group.spec.fp16_tflops}T</span>
          <span className="chip">{group.spec.nvlink ? "NVLink" : "无高速互联"}</span>
          <span
            className={
              "chip " +
              (group.spec.source === "datasheet"
                ? "text-emerald-400"
                : group.spec.source.startsWith("measured")
                ? "text-sky-400"
                : "text-amber-400")
            }
            title={group.spec.note}
          >
            {group.spec.source === "datasheet"
              ? "✓ 官方规格"
              : group.spec.source.startsWith("measured")
              ? "✓ 实测"
              : "⚠ 估值存疑"}
          </span>
        </div>
        {group.spec.note && (
          <p className="mt-1 text-[11px] leading-snug text-slate-300">{group.spec.note}</p>
        )}
      </div>
      <p className="mt-2 text-[11px] text-slate-300">
        张量并行 (TP) 部署，单一卡型 × 数量。
      </p>
    </div>
  );
}
