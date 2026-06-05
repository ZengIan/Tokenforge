import { useStore } from "../store";
import type { EstimateResponse } from "../types";
import { Tip } from "./Tip";

export function ResultPanel() {
  const { result, loading, error } = useStore();

  const heading = (
    <h2 className="mb-1 flex items-center gap-2 text-sm font-bold text-forge-flame">
      性能测算结果
      <span className="text-[11px] font-normal text-slate-300">
        （估算为理论值，实际推理性能请以实测数据为准）
      </span>
    </h2>
  );

  if (error) {
    return (
      <div className="space-y-1">
        {heading}
        <div className="card text-amber-400">估算失败：{error}</div>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="space-y-1">
        {heading}
        <div className="card flex h-40 items-center justify-center text-slate-300">
          {loading ? "锻造中… 🔥" : "调整参数即可自动估算"}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {heading}
      {!result.analysis_reliable && (
        <div className="rounded-lg border border-amber-700/60 bg-amber-950/30 px-3 py-2 text-[11px] leading-snug text-amber-300">
          ⚠ 所选 GPU 为占位/估值规格（算力或带宽未填真实值），下方 <b>TPS、延迟、瓶颈分析不可靠</b>，
          仅 <b>显存占用 / 可容纳并发</b> 可参考。请在 <code>gpus.yaml</code> 填入厂商规格书或实测值
          （把 <code>source</code> 改为 <code>datasheet</code>/<code>measured</code>）后再看性能结论。
        </div>
      )}
      {/* 第一行：性能测算指标卡 */}
      <MetricCards r={result} />
      {/* 第二行：显存分解 */}
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

interface Metric {
  label: string;
  value: string;
  unit: string;
  tip: string;
  accent?: boolean;
}

function MetricCards({ r }: { r: EstimateResponse }) {
  const cards: Metric[] = [
    {
      label: "TPS 总吞吐",
      value: range(r.tps_low, r.tps_high),
      unit: "tokens/s",
      tip:
        "保守区间(非理论上限) = 并发数 × 单请求 TPS。\n" +
        "单 token 耗时 = 权重读取(带宽)时间 + 每 token 固定开销。\n" +
        "区间反映 kernel/通信开销的不确定性，已按实测标定。\n" +
        "并发越大总吞吐越高(直到触及算力/带宽上限)。",
    },
    {
      label: "单请求 TPS",
      value: range(r.single_tps_low, r.single_tps_high),
      unit: "tokens/s",
      tip:
        "单条请求每秒生成的 token 数(保守区间，非理论上限)。\n" +
        "= 1 ÷ 单 token 耗时；单 token 耗时 = 权重读取时间 + 固定开销。\n" +
        "固定开销含 kernel 启动、TP all-reduce 延迟等，\n" +
        "在 batch=1 / enforce-eager / MoE·线性注意力 下会显著拉低。\n" +
        "区间已用真实部署数据标定，比纯屋顶线更贴近实测。",
    },
    {
      label: "可容纳并发",
      value: fmt(r.max_fit_seqs),
      unit: "seqs",
      tip:
        "= (显存预算 − 权重 − 激活 − 框架开销) ÷ 每路 KV 显存\n" +
        "显存预算 = 单卡显存 × gpu-memory-utilization × 卡数。\n" +
        "表示预算内最多能同时容纳多少路并发。",
    },
    {
      label: "TTFT 首字延迟",
      value: r.ttft_ms.toFixed(0),
      unit: "ms",
      tip:
        "模型把整段问题读一遍(prefill)的时间，决定“第一个字”多久蹦出来。\n" +
        "= (2 × 激活参数 × 输入长度 + 注意力 O(n²)) ÷ (算力 × 利用率) + 固定开销\n" +
        "★ 随『输入长度(Prompt)』线性增长，长上下文还有 O(n²) 项。\n" +
        "短输入(几百~2K)多卡下就是几十~几百 ms；长输入(8K~32K)会到秒级。",
    },
    {
      label: "TPOT 每字延迟",
      value: r.tpot_ms.toFixed(2),
      unit: "ms",
      tip:
        "每生成 1 个 token 的耗时(中值)。\n" +
        "= 权重读取时间 + 每 token 固定开销\n" +
        "• 权重读取 = 激活权重字节 ÷ (总带宽 × 利用率)\n" +
        "• 固定开销 = 层数 × 每层开销 ×(MoE/线性/TP/eager 系数)\n" +
        "batch=1 时固定开销常是主导，所以单请求远低于带宽上限。",
    },
    {
      label: "总显存占用",
      value: r.memory.total_gb.toFixed(2),
      unit: "GB",
      tip:
        "这套配置的真实显存需求 = 模型权重 + KV Cache + 激活值 + 框架开销\n" +
        "(KV 按 max-model-len × max-num-seqs 满算)。\n" +
        "可能超过物理显存——超了说明放不下全部并发，vLLM 会自动降并发(见“可容纳并发”)。",
    },
    {
      label: "单卡占用",
      value: r.memory.per_gpu_gb.toFixed(2),
      unit: "GB",
      tip:
        "= (权重 + KV) ÷ 卡数 + 激活 ÷ 卡数 + 每卡固定开销\n" +
        "张量并行(TP)把权重和 KV 均摊到每张卡。",
    },
    {
      label: "显存利用率",
      value: (r.mem_utilization * 100).toFixed(0),
      unit: "%",
      tip:
        "= 显存需求 ÷ 显存预算(总显存 × gpu-memory-utilization)\n" +
        "<100%：放得下，还有余量；\n" +
        "=100%：刚好占满预算；\n" +
        ">100%：放不下全部请求，vLLM 会把并发降到“可容纳并发”路。",
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
      {cards.map((c) => (
        <div key={c.label} className="card relative flex flex-col justify-center p-3" style={{ borderColor: 'rgba(240, 237, 235, 0.4)' }}>
          <div className="flex items-center gap-1 text-[11px] text-slate-300">
            {c.label}
            <Tip text={c.tip} />
          </div>
          <div className="mt-1 flex items-baseline gap-1 whitespace-nowrap text-slate-100">
            <span className="text-lg font-bold leading-tight">{c.value}</span>
            <span className="text-[10px] font-normal text-slate-400">{c.unit}</span>
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

const MEM_FORMULA =
  "显存需求由四部分构成（按满并发满上下文的真实需求，不截断）：\n" +
  "• 模型权重：有官方大小用官方；否则 参数量 × 精度字节数 × 量化开销\n" +
  "• KV Cache = 2 × 层数 × KV维度 × 上下文长度 × 并发数 × KV字节 ÷ 0.9\n" +
  "   (线性/混合注意力按 kv_cache_factor 缩减)\n" +
  "• 激活值 ≈ max-num-batched-tokens × hidden_size × 字节 × 2\n" +
  "• 框架开销 ≈ 每卡 1.2GB（CUDA 上下文等），enforce-eager 省约 0.6GB/卡\n\n" +
  "若需求 > 显存预算，vLLM 会自动降低并发(见“可容纳并发”)，不是直接 OOM。";

function MemoryBreakdownChart({ r }: { r: EstimateResponse }) {
  const total = r.memory.total_gb || 1;
  const over = !r.fits; // 需求超过预算 → 会降并发
  const budget = r.mem_utilization > 0 ? r.memory.total_gb / r.mem_utilization : r.total_mem_gb;
  return (
    <div className="card relative">
      <div className="mb-1 flex items-center">
        <h3 className="flex items-center gap-1 text-xs font-bold text-slate-300">
          显存分解
          <Tip text={MEM_FORMULA} />
        </h3>
        {over && (
          <span className="hidden flex-1 text-center text-[11px] text-amber-300 lg:inline">
            需求 {r.memory.total_gb.toFixed(0)}GB 超过预算，实际并发会降到约 {r.memory.max_kv_seqs} 路
          </span>
        )}
        <span className={"ml-auto text-xs " + (r.fits ? "text-emerald-400" : "text-amber-400")}>
          {r.fits
            ? `✓ 可容纳 · 利用 ${(r.mem_utilization * 100).toFixed(0)}%`
            : `⚠ 超预算 · 实际约 ${r.memory.max_kv_seqs} 路`}{" "}
          · 预算 {budget.toFixed(0)}GB
        </span>
      </div>
      <div className="flex h-5 w-full overflow-hidden rounded-lg bg-slate-900">
        {SEGMENTS.map((s) => {
          const v = r.memory[s.key];
          const pct = (v / total) * 100;
          return pct > 0.5 ? (
            <div
              key={s.key}
              className={s.color}
              style={{ width: `${pct}%` }}
              title={`${s.label}: ${v.toFixed(2)}GB`}
            />
          ) : null;
        })}
      </div>
      <div className="mt-1.5 grid grid-cols-2 gap-1 text-[11px] sm:grid-cols-4">
        {SEGMENTS.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5">
            <span className={"inline-block h-2 w-2 rounded-sm " + s.color} />
            <span className="text-slate-300">{s.label}</span>
            <span className="text-slate-200">{r.memory[s.key].toFixed(2)}GB</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const BOTTLENECK_LABEL: Record<string, string> = {
  "Memory Bound": "显存瓶颈",
  "Bandwidth Bound": "带宽瓶颈",
  "Compute Bound": "算力瓶颈",
};

const COEF_TIP =
  '这两个系数表示"实际能用到的比例"，因为纸面峰值跑不满：\n\n' +
  "• 算力利用率：真实算力 ÷ 显卡标称算力。受 kernel 效率、TP 通信、" +
  "小 batch 等影响，vLLM 经验值约 60%。\n" +
  "• 带宽利用率：真实显存读写速度 ÷ 标称带宽。decode 主要靠它，约 80%。\n\n" +
  "数值越高代表越接近硬件极限。可在高级参数里手动覆盖来贴合你的实测。";

function Bottleneck({ r }: { r: EstimateResponse }) {
  const tag = BOTTLENECK_LABEL[r.bottleneck] || r.bottleneck;
  return (
    <div className="card relative border-slate-600/60 text-slate-300">
      <h3 className="flex flex-wrap items-center gap-1.5 text-xs font-bold text-forge-flame">
        优化建议
        {r.analysis_reliable ? (
          <span className="chip border-slate-600 text-[10px] text-slate-300">
            当前瓶颈 · {tag}
          </span>
        ) : (
          <span className="chip border-amber-700/60 text-[10px] text-amber-300">
            参数存疑 · 仅供参考
          </span>
        )}
        <span className="ml-auto flex items-center gap-1 text-[10px] text-slate-300">
          计算经验值(影响最终计算结果)：算力利用率 {(r.effective_compute_util * 100).toFixed(0)}% · 带宽利用率{" "}
          {(r.effective_mem_util * 100).toFixed(0)}%
          <Tip text={COEF_TIP} />
        </span>
      </h3>
      <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] text-slate-200">
        {r.suggestions.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
    </div>
  );
}

function fmt(n: number) {
  return n >= 1000 ? Math.round(n).toLocaleString() : n.toFixed(2);
}

/** 整数化的区间显示，如 "14–30" */
function range(low: number, high: number) {
  const a = Math.round(low);
  const b = Math.round(high);
  return a === b ? `${a}` : `${a}–${b}`;
}
