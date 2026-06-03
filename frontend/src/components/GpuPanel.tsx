import { useStore } from "../store";

const PRESETS: { label: string; gpu: string; count: number }[] = [
  { label: "8×H20-3e", gpu: "H20-3e (141G)", count: 8 },
  { label: "4×H100", gpu: "H100 SXM (80G)", count: 4 },
  { label: "8×A100", gpu: "A100 SXM (80G)", count: 8 },
  { label: "8×910C", gpu: "Ascend 910C (64G)", count: 8 },
];

export function GpuPanel() {
  const { gpuDb, gpuGroups, addGpuGroup, updateGpuGroup, removeGpuGroup, setGpuDb } =
    useStore();

  function applyPreset(p: { gpu: string; count: number }) {
    const spec = gpuDb.find((g) => g.name === p.gpu);
    if (!spec) return;
    setGpuDb(gpuDb); // no-op keeps types happy
    useStore.setState({ gpuGroups: [{ spec, count: p.count }] });
  }

  return (
    <div className="card">
      <h2 className="mb-3 text-sm font-bold text-forge-flame">② GPU 配置</h2>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button key={p.label} className="chip hover:border-forge-ember" onClick={() => applyPreset(p)}>
            {p.label}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        {gpuGroups.map((g, idx) => (
          <div key={idx} className="rounded-lg border border-slate-700 bg-slate-900/50 p-2">
            <div className="flex items-center gap-2">
              <select
                className="input flex-1"
                value={g.spec.name}
                onChange={(e) => {
                  const spec = gpuDb.find((x) => x.name === e.target.value);
                  if (spec) updateGpuGroup(idx, { spec });
                }}
              >
                {gpuDb.map((x) => (
                  <option key={x.name} value={x.name}>
                    {x.name}
                  </option>
                ))}
              </select>
              <input
                type="number"
                min={1}
                className="input w-16"
                value={g.count}
                onChange={(e) => updateGpuGroup(idx, { count: Math.max(1, Number(e.target.value)) })}
              />
              <button
                className="px-2 text-slate-500 hover:text-red-400"
                onClick={() => removeGpuGroup(idx)}
                title="删除"
              >
                ✕
              </button>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px] text-slate-300">
              <span className="chip">{g.spec.mem_gb}GB</span>
              <span className="chip">{g.spec.bw_gbs} GB/s</span>
              <span className="chip">FP16 {g.spec.fp16_tflops}T</span>
              <span className="chip">{g.spec.nvlink ? "NVLink" : "无高速互联"}</span>
              <span
                className={
                  "chip " +
                  (g.spec.source === "datasheet"
                    ? "text-emerald-400"
                    : g.spec.source.startsWith("measured")
                    ? "text-sky-400"
                    : "text-amber-400")
                }
                title={g.spec.note}
              >
                {g.spec.source === "datasheet"
                  ? "✓ 官方规格"
                  : g.spec.source.startsWith("measured")
                  ? "✓ 实测"
                  : "⚠ 估值存疑"}
              </span>
            </div>
            {g.spec.note && (
              <p className="mt-1 text-[11px] leading-snug text-slate-300">{g.spec.note}</p>
            )}
          </div>
        ))}
      </div>

      <button
        className="mt-2 w-full rounded-lg border border-dashed border-slate-600 py-1.5 text-sm text-slate-300 hover:border-forge-ember hover:text-forge-ember"
        onClick={() => gpuDb[0] && addGpuGroup(gpuDb[0])}
      >
        + 添加 GPU（支持异构混装）
      </button>
    </div>
  );
}
