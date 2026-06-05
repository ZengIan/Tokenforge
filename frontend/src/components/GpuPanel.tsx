import { useStore } from "../store";
import type { GpuSpec, IntraNode } from "../types";

/** 根据卡型名称返回高速互联技术名称 */
function describeInterconnect(spec: GpuSpec): string {
  if (!spec.nvlink) return "无高速互联";
  const n = spec.name;
  if (n.includes("NVIDIA")) return "NVLink";
  if (n.includes("华为") || n.includes("Ascend") || n.includes("昇腾")) return "HCCS";
  if (n.includes("海光") || n.includes("Hygon") || n.includes("DCU")) return "xGMI";
  if (n.includes("平头哥") || n.includes("PPU")) return "ICN";
  return "高速互联";
}

export function GpuPanel() {
  const { gpuDb, gpuGroups, updateGpuGroup, inference, setInference } = useStore();
  const group = gpuGroups[0];
  const nGpu = gpuGroups.reduce((s, g) => s + g.count, 0);
  const isSingle = inference.gpus_per_node >= nGpu && nGpu > 0;

  function setGpu(name: string) {
    const spec = gpuDb.find((x) => x.name === name);
    if (spec) updateGpuGroup(0, { spec });
  }

  function toggleSingle(checked: boolean) {
    setInference({ gpus_per_node: checked ? nGpu : 8 });
  }

  if (!group) {
    return <div className="card text-slate-300">加载 GPU 卡库中…</div>;
  }

  return (
    <div className="card">
      <div className="mb-3 flex items-center gap-8">
        <h2 className="text-sm font-bold text-forge-flame">② GPU 配置</h2>
        <div className="flex gap-2.5">
          {[1, 2, 4, 8, 16, 32].map((n) => (
            <button
              key={n}
              className={
                "rounded-md border border-slate-600 bg-slate-700/50 px-4 py-1.5 text-[10px] " +
                (group.count === n ? "border-forge-ember text-forge-ember" : "")
              }
              onClick={() => updateGpuGroup(0, { count: n })}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-2">
        <div className="flex items-center gap-2">
          <select
            className="input flex-[3] min-w-0"
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
          <label
            className="flex cursor-pointer items-center gap-1 text-[11px] text-slate-300"
            title={isSingle ? "当前视为单机部署，所有卡在同一节点内" : `超过 ${inference.gpus_per_node} 卡即视为多机，触发跨机通信损耗`}
          >
            <input
              type="checkbox"
              className="accent-forge-ember"
              checked={isSingle}
              onChange={(e) => toggleSingle(e.target.checked)}
            />
            单机
          </label>
          {nGpu > 1 && (
            <select
              className="input w-36 shrink-0 text-xs"
              value={inference.intra_node}
              onChange={(e) => setInference({ intra_node: e.target.value as IntraNode })}
              title="单机内卡间互联方式：影响张量并行(TP)通信效率。高速 NVLink/HCCS 几乎无损，纯 PCIe 约 -15%。"
            >
              <option value="auto">机内：按卡库识别</option>
              <option value="highspeed">NVLink / HCCS</option>
              <option value="pcie">纯 PCIe</option>
            </select>
          )}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px] text-slate-300">
          <span className="chip">{group.spec.mem_gb}GB</span>
          <span className="chip">{group.spec.bw_gbs} GB/s</span>
          <span className="chip">FP16 {group.spec.fp16_tflops}T</span>
          <span className="chip">{describeInterconnect(group.spec)}</span>
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
          {nGpu > inference.gpus_per_node && (
            <span className="chip border-amber-700/60 text-amber-300">
              多机 ({Math.ceil(nGpu / inference.gpus_per_node)} 节点) · {inference.gpus_per_node} 卡/节点
            </span>
          )}
        </div>
        {group.spec.note && (
          <p className="mt-1 text-[11px] leading-snug text-slate-300">{group.spec.note}</p>
        )}
      </div>
      <p className="mt-2 text-[11px] text-slate-300">
        张量并行 (TP) 部署，单一卡型 × 数量。
        {nGpu > 1 && " 右侧「?」提示查看互联影响。"}
      </p>
    </div>
  );
}
