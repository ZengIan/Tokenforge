import type {
  EstimateResponse,
  GpuGroup,
  InferenceConfig,
  ModelSpec,
} from "../types";

/** 一条已加入清单的测算记录(快照,导出时不受后续改动影响) */
export interface ExportRecord {
  id: string;
  label: string; // 列表展示用
  cells: (string | number)[]; // 与 COLUMNS 对齐
}

export const COLUMNS = [
  "算力卡类型",
  "模型",
  "权重大小(GB)",
  "参数量(B)",
  "量化类型",
  "GPU配置(卡)",
  "上下文(k)",
  "期望并发数(个)",
  "KV cache (GB)",
  "激活值(GB)",
  "框架开销(GB)",
  "总显存开销(权重+激活+KV+框架)",
  "可容纳并发数(个)",
  "单用户 TPS(tokens/s)",
  "首字延迟 TTFT(ms)",
  "TPOT 每字延迟(ms)",
  "体感评级",
  "说明",
];

/** 体感评级:依据单用户 TPS(取保守下限) 与 TTFT(秒) */
function rating(tps: number, ttftS: number): [string, string] {
  if (tps >= 30 && ttftS <= 2) return ["✅ 流畅", "对话流畅，体感接近公共模型商用服务"];
  if (tps >= 20 && ttftS <= 5) return ["✅ 良好", "对话流畅，长输入完全可接受"];
  if (tps >= 15 && ttftS <= 10) return ["🟢 可接受", "略慢但完全可用，适合内部生产"];
  if (tps >= 10 && ttftS <= 20) return ["🟡 偏慢", "首字慢，吐字尚可，建议优化或限并发"];
  if (tps >= 5 || ttftS <= 40) return ["🟠 仅演示", "生产慎用，仅适合演示/批处理"];
  return ["🔴 离线/批处理", "体感差，建议降配置或扩资源"];
}

const n2 = (v: number) => Math.round(v * 100) / 100;

export function buildRecord(
  model: ModelSpec,
  gpuGroups: GpuGroup[],
  inf: InferenceConfig,
  r: EstimateResponse
): ExportRecord {
  const cardName = gpuGroups[0]?.spec.name ?? "";
  const nGpu = gpuGroups.reduce((s, g) => s + g.count, 0);
  const ctxK = Math.round(inf.max_model_len / 1024);
  const tpsLow = Math.round(r.single_tps_low);
  const tpsHigh = Math.round(r.single_tps_high);
  const [grade, desc] = rating(r.single_tps_low, r.ttft_ms / 1000);

  const cells: (string | number)[] = [
    cardName,
    model.model_id,
    n2(r.memory.weights_gb),
    model.params_b,
    inf.quantization,
    nGpu,
    ctxK,
    inf.max_num_seqs,
    n2(r.memory.kv_cache_gb),
    n2(r.memory.activations_gb),
    n2(r.memory.overhead_gb),
    n2(r.memory.total_gb),
    r.max_fit_seqs,
    tpsLow === tpsHigh ? `${tpsLow}` : `${tpsLow}-${tpsHigh}`,
    Math.round(r.ttft_ms),
    n2(r.tpot_ms),
    grade,
    desc,
  ];
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    label: `${cardName} ×${nGpu} · ${model.model_id} · ${ctxK}K/${inf.max_num_seqs}路`,
    cells,
  };
}

/** 测算依据(公式说明),覆盖表格每个指标 + 补充项;附在 CSV 末尾 */
const BASIS: [string, string][] = [
  ["权重大小(GB)", "运行时权重显存 = 参数量 × 每参数字节数(量化决定:BF16=2 / FP8·W8=1 / 4bit=0.5) × 组量化开销。MoE 所有专家常驻显存,按总参数算。"],
  ["KV cache(GB)", "= 2 × 层数 × KV维度 × 上下文长度 × 并发数 × KV字节 ÷ 0.9(分页)。KV维度=KV头数×head_dim(GQA);线性/混合注意力按 kv_cache_factor 缩减(KV 极小)。显示为满并发满上下文的真实需求。"],
  ["激活值(GB)", "≈ max-num-batched-tokens × hidden_size × 精度字节 × 2(峰值中间张量经验系数)。"],
  ["框架开销(GB)", "≈ 每卡约 1.2GB(CUDA 上下文/算子 workspace/通信缓冲) × 卡数;开启 enforce-eager 每卡省约 0.6GB。"],
  ["总显存开销(GB)", "= 权重 + KV cache + 激活值 + 框架开销(真实需求)。若 > 卡数×单卡显存×gpu-memory-utilization,则放不下全部并发,vLLM 自动降并发(见可容纳并发数)。"],
  ["可容纳并发数(个)", "= (显存预算 − 权重 − 激活 − 框架开销) ÷ 每路 KV 显存;显存预算 = 卡数 × 单卡显存 × gpu-memory-utilization。表示预算内实际能同时跑多少路。"],
  ["单用户 TPS(tokens/s)", "保守区间 = 1 ÷ 单 token 耗时;单 token 耗时 = 激活权重 ÷ (聚合带宽×带宽利用率) + 每 token 固定开销(层数×每层开销×MoE/线性/TP/eager 系数)。区间已按真实部署实测标定。"],
  ["首字延迟 TTFT(ms)", "= 2 × 激活参数量 × max-num-batched-tokens ÷ (聚合算力 × 算力利用率)。即 prefill(读题)时间,MoE 按激活参数算。"],
  ["TPOT 每字延迟(ms)", "= 单 token 耗时(权重读取时间 + 每 token 固定开销)。batch=1 时固定开销常占主导,故单请求远低于带宽上限。"],
  ["效率系数", "算力利用率≈真实算力÷标称算力(受 kernel/通信影响,vLLM 约 60%);带宽利用率≈真实读写÷标称带宽(decode 主要靠它,约 80%)。"],
  ["跨机互联损耗", ">8 卡视为多机:高速无损(NVLink Switch/HCCS)≈无损;InfiniBand/RoCE 约 -10%;普通以太网约 -25%,且每 token 跨机延迟更高。单机内无 NVLink(PCIe) 约 -15%。"],
  ["量化类型", "fp8/w8a8=8bit(1B/参数);awq/gptq/w4a8/w4a16=4bit(0.5B/参数);none=按 dtype。量化越激进越省显存,精度损失越大。"],
  ["体感评级阈值", "✅流畅 TPS≥30且TTFT≤2s;✅良好 TPS≥20且TTFT≤5s;🟢可接受 TPS≥15且TTFT≤10s;🟡偏慢 TPS≥10且TTFT≤20s;🟠仅演示 TPS≥5或TTFT≤40s;🔴其他为离线/批处理。TPS 取单用户保守下限。"],
  ["说明", "估算为理论近似,实际推理性能以实测为准;国产卡 source=estimate 的算力/带宽为占位值,性能仅供参考。"],
];

function csvCell(v: string | number): string {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function csvRow(cells: (string | number)[]): string {
  return cells.map(csvCell).join(",");
}

/** 生成完整 CSV 文本:性能矩阵 + 测算依据 */
export function recordsToCsv(records: ExportRecord[]): string {
  const lines: string[] = [];
  lines.push(csvRow(COLUMNS));
  for (const r of records) lines.push(csvRow(r.cells));
  // 空行分隔 + 测算依据
  lines.push("");
  lines.push(csvRow(["测算依据(公式与口径)", ""]));
  lines.push(csvRow(["指标", "说明"]));
  for (const [k, v] of BASIS) lines.push(csvRow([k, v]));
  return lines.join("\r\n");
}

/** 触发浏览器下载 */
export function downloadCsv(records: ExportRecord[], filename = "tokenforge-性能测算.csv") {
  // 加 BOM 以便 Excel 正确识别 UTF-8 中文
  const blob = new Blob(["﻿" + recordsToCsv(records)], {
    type: "text/csv;charset=utf-8;",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
