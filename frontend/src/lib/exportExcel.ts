import ExcelJS from "exceljs";
import type { EstimateResponse, GpuGroup, InferenceConfig, ModelSpec } from "../types";

export interface ExportRecord {
  id: string;
  label: string;
  cells: (string | number)[];
}

/* ---------- 样式常量 ---------- */
const ORG = "FF1F4E78";
const HEADER_FILL: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: ORG } };
const BODY_FILL: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF7FBFE" } };
const GRAY_FILL: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FFF2F2F2" } };
const BORDER: Partial<ExcelJS.Borders> = {
  top: { style: "thin", color: { argb: "FFBFBFBF" } },
  bottom: { style: "thin", color: { argb: "FFBFBFBF" } },
  left: { style: "thin", color: { argb: "FFBFBFBF" } },
  right: { style: "thin", color: { argb: "FFBFBFBF" } },
};
const CENTER: Partial<ExcelJS.Alignment> = { vertical: "middle", horizontal: "center", wrapText: true };
const LEFT: Partial<ExcelJS.Alignment> = { vertical: "middle", horizontal: "left", wrapText: true };

/* ---------- 字体 ---------- */
const F10 = (b?: boolean): Partial<ExcelJS.Font> => ({ name: "微软雅黑", size: 10, bold: b ?? false });
const F11B = (): Partial<ExcelJS.Font> => ({ name: "微软雅黑", bold: true, size: 11 });
const F11W = (): Partial<ExcelJS.Font> => ({ name: "微软雅黑", bold: true, color: { argb: "FFFFFFFF" }, size: 11 });
const F10W = (): Partial<ExcelJS.Font> => ({ name: "微软雅黑", bold: true, color: { argb: "FFFFFFFF" }, size: 10 });
const F14W = (): Partial<ExcelJS.Font> => ({ name: "微软雅黑", bold: true, color: { argb: "FFFFFFFF" }, size: 14 });

/* ---------- 体感评级 ---------- */
const RATING_MAP: Record<string, { fill: string; grade: string }> = {
  流畅: { fill: "FFD5F5DA", grade: "✅ 流畅" },
  良好: { fill: "FFE2F5DA", grade: "✅ 良好" },
  可接受: { fill: "FFEAF7E6", grade: "🟢 可接受" },
  偏慢: { fill: "FFFFF4D6", grade: "🟡 偏慢" },
  "仅演示": { fill: "FFFFE8D1", grade: "🟠 仅演示" },
  "离线/批处理": { fill: "FFFFD6D6", grade: "🔴 不可用" },
};

const RATING_ITEMS = [
  ["✅ 流畅", "TPS ≥ 30 且 TTFT ≤ 2s", "D5F5DA", "对话流畅，体感接近公共模型商用服务"],
  ["✅ 良好", "TPS ≥ 20 且 TTFT ≤ 5s", "E2F5DA", "对话流畅，长输入完全可接受"],
  ["🟢 可接受", "TPS ≥ 15 且 TTFT ≤ 10s", "EAF7E6", "略慢但完全可用，适合内部生产"],
  ["🟡 偏慢", "TPS ≥ 10 且 TTFT ≤ 20s", "FFF4D6", "首字慢，吐字尚可，建议优化或限并发"],
  ["🟠 仅演示", "TPS ≥ 5 或 TTFT ≤ 40s", "FFE8D1", "生产慎用，仅适合演示/批处理"],
  ["🔴 不可用", "其他", "FFD6D6", "体感差，建议降配置或扩资源"],
] as const;

function grade(tps: number, ttftMs: number): string {
  const s = ttftMs / 1000;
  if (tps >= 30 && s <= 2) return "流畅";
  if (tps >= 20 && s <= 5) return "良好";
  if (tps >= 15 && s <= 10) return "可接受";
  if (tps >= 10 && s <= 20) return "偏慢";
  if (tps >= 5 || s <= 40) return "仅演示";
  return "离线/批处理";
}

const n2 = (v: number) => Math.round(v * 100) / 100;

/* ---------- 性能估算列 ---------- */
const COLS = [
  { h: "算力卡类型", w: 18 },
  { h: "模型", w: 22 },
  { h: "参数量(B)", w: 13 },
  { h: "量化类型", w: 12 },
  { h: "上下文(K)", w: 13 },
  { h: "权重大小(GB)", w: 14 },
  { h: "KV Cache(GB)", w: 13 },
  { h: "激活值(GB)", w: 11 },
  { h: "框架开销(GB)", w: 13 },
  { h: "总显存(GB)", w: 13 },
  { h: "预算(GB)", w: 12 },
  { h: "可容纳并发", w: 12 },
  { h: "单用户TPS", w: 13 },
  { h: "TTFT(ms)", w: 10 },
  { h: "TPOT(ms)", w: 10 },
  { h: "体感评级", w: 12 },
  { h: "说明", w: 48 },
];
const N = COLS.length;

export function buildRecord(
  model: ModelSpec,
  gpuGroups: GpuGroup[],
  inf: InferenceConfig,
  r: EstimateResponse,
): ExportRecord {
  const card = gpuGroups[0]?.spec.name ?? "";
  const nGpu = gpuGroups.reduce((s, g) => s + g.count, 0);
  const ctxK = Math.round(inf.max_model_len / 1024);
  const g = grade(r.single_tps_low, r.ttft_ms);
  const gv = RATING_MAP[g] || { fill: "FFFFFFFF", grade: g };
  const budget = r.mem_utilization > 0 ? r.memory.total_gb / r.mem_utilization : r.total_mem_gb;
  const tpsLow = Math.round(r.single_tps_low);
  const tpsHigh = Math.round(r.single_tps_high);

  const cells: (string | number)[] = [
    card, model.model_id, model.params_b, inf.quantization, ctxK,
    n2(r.memory.weights_gb), n2(r.memory.kv_cache_gb), n2(r.memory.activations_gb),
    n2(r.memory.overhead_gb), n2(r.memory.total_gb), n2(budget), r.max_fit_seqs,
    tpsLow === tpsHigh ? `${tpsLow}` : `${tpsLow}–${tpsHigh}`,
    Math.round(r.ttft_ms), n2(r.tpot_ms),
    gv.grade,
    gv.grade === "✅ 流畅" ? "对话流畅，体感接近公共模型商用服务" :
    gv.grade === "✅ 良好" ? "对话流畅，长输入完全可接受" :
    gv.grade === "🟢 可接受" ? "略慢但完全可用，适合内部生产" :
    gv.grade === "🟡 偏慢" ? "首字慢，吐字尚可，建议优化或限并发" :
    gv.grade === "🟠 仅演示" ? "生产慎用，仅适合演示/批处理" : "体感差，建议降配置或扩资源",
  ];
  return { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, label: `${card} ×${nGpu}`, cells };
}

/* ========== 生成 ========== */
export async function downloadExcel(records: ExportRecord[], filename = "模型推理性能估算.xlsx") {
  const wb = new ExcelJS.Workbook();

  /* ---- Sheet 1: 测算依据 ---- */
  const s1 = wb.addWorksheet("测算依据");
  s1.getColumn(1).width = 10;
  s1.getColumn(2).width = 58;
  s1.getColumn(3).width = 48;

  const h1 = s1.addRow(["指标", "说明", "示例： Ascend 910C (128G) × 4 卡，DeepSeek-V4-Flash (300.1B 参数, w8a8) 200K / 8路"]);
  h1.height = 32;
  h1.getCell(1).font = F11W(); h1.getCell(1).fill = HEADER_FILL; h1.getCell(1).alignment = CENTER;
  h1.getCell(2).font = F11W(); h1.getCell(2).fill = HEADER_FILL; h1.getCell(2).alignment = CENTER;
  h1.getCell(3).font = F10W(); h1.getCell(3).fill = HEADER_FILL; h1.getCell(3).alignment = CENTER;

  const items: [string, string, string][] = [
    ["权重大小", "运行时权重显存 = 参数量 × 每参数字节数(量化决定:BF16=2 / FP8·W8=1 / 4bit=0.5) × 组量化开销。MoE 所有专家常驻显存,按总参数算。", "300.1B × 1 字节(w8a8) = 300.1 GB"],
    ["KV Cache", "= 2 × 层数 × KV维度 × 上下文长度 × 并发数 × KV字节 ÷ 0.9(分页)。KV维度=KV头数×head_dim(GQA);\n线性/混合注意力按 kv_cache_factor 缩减(KV 极小)。显示为满并发满上下文的真实需求。", "KV 维度 = 1 (MLA)×512(head_dim)=512\n标准公式:2×43×512×200K×8×1÷0.9\nMLA 压缩后 ≈ 155 GB"],
    ["激活值", "≈ max-num-batched-tokens × hidden_size × 字节 × 2(峰值中间张量经验系数)", "2048 × 7168 × 2 × 2 ≈ 0.03 GB"],
    ["框架开销", "≈ 每卡约 1.2GB(CUDA 上下文/算子 workspace/通信缓冲) × 卡数,\n开启 enforce-eager 每卡省约 0.6GB。", "1.2 GB × 4 卡 = 4.8 GB"],
    ["总显存开销", "= 权重 + KV cache + 激活值 + 框架开销(真实需求)。\n若 > 卡数×单卡显存×gpu-memory-utilization,则放不下全部并发,\nvLLM 自动降并发(见可容纳并发数)。", "300.1 + 155 + 0.03 + 4.8 = 459.93 GB"],
    ["可容纳并发数", "= (显存预算 − 权重 − 激活 − 框架开销) ÷ 每路 KV 显存;\n显存预算 = 单卡显存 × gpu-memory-utilization × 卡数。\n表示预算内实际能同时跑多少路。", "预算 = 4 × 128 × 0.9 = 461 GB\n每路 KV = 155 ÷ 8 ≈ 19.4 GB\n可容纳 ≈ (461−300−0.03−4.8)÷19.4 ≈ 7 路"],
    ["单用户 TPS", "保守区间 = 1 ÷ 单 token 耗时;\n单 token 耗时 = 权重读取时间 + 每 token 固定开销。\n区间已用真实部署数据标定。", "71-140 tokens/s"],
    ["首字延迟 TTFT", "= 2 × 激活参数量 × max-num-batched-tokens ÷ (聚合算力 × 算力利用率)。\n即 prefill(读题)时间,MoE 按激活参数算。", "28 ms"],
    ["TPOT 每字延迟", "= 单 token 耗时(权重读取时间 + 每 token 固定开销)。\nbatch=1 时固定开销常占主导,故单请求远低于带宽上限。", "9.15 ms"],
    ["效率系数", "算力利用率≈真实算力÷标称算力(受 kernel/通信影响,vLLM 约 60%);\n带宽利用率≈真实读写÷标称带宽(decode 主要靠它,约 80%)。", "算力 60% · 带宽 80%"],
    ["跨机互联损耗", ">8 卡视为多机:高速无损(NVLink Switch/HCCS)≈无损;\nInfiniBand/RoCE 约 -10%;普通以太网约 -25%;\n单机内无 NVLink(PCIe) 约 -15%。", "910C 单机 4 卡 ≈ -15%"],
    ["量化类型", "fp8/w8a8=8bit(1B/参数);awq/gptq/w4a8/w4a16=4bit(0.5B/参数);\nnone=按 dtype。量化越激进越省显存,精度损失越大。", "w8a8 = 1 字节/参数"],
    ["体感评级阈值", "✅流畅 TPS≥30且TTFT≤2s;✅良好 TPS≥20且TTFT≤5s;\n🟢可接受 TPS≥15且TTFT≤10s;🟡偏慢 TPS≥10且TTFT≤20s;\n🟠仅演示 TPS≥5或TTFT≤40s;🔴其他为离线/批处理。\nTPS 取单用户保守下限。", "当前: ✅ 流畅"],
    ["说明", "估算为理论近似,实际推理性能以实测为准;\n国产卡 source=estimate 的算力/带宽为占位值,性能仅供参考。", "910C 无原生 FP8,已回退到 FP16 算力估算"],
  ];

  for (const [name, desc, example] of items) {
    const r = s1.addRow([name, desc, example]);
    const lines = desc.split("\n").length;
    r.height = Math.max(40, lines * 16);
    r.getCell(1).font = F10(true); r.getCell(1).alignment = CENTER;
    r.getCell(2).font = F10(false); r.getCell(2).alignment = LEFT; r.getCell(2).fill = BODY_FILL;
    r.getCell(3).font = F10(false); r.getCell(3).alignment = LEFT;
  }

  /* ---- Sheet 2: 性能估算 ---- */
  const s2 = wb.addWorksheet("性能估算", { views: [{ state: "frozen", ySplit: 1 }] });
  COLS.forEach((c, i) => { s2.getColumn(i + 1).width = c.w; });

  // 标题行
  const title = s2.addRow(["🚀 模型推理性能矩阵 · 按上下文长度递增、并发递减"]);
  title.height = 32;
  title.getCell(1).font = F14W();
  title.getCell(1).fill = HEADER_FILL;
  title.getCell(1).alignment = CENTER;
  s2.mergeCells(1, 1, 1, N);
  for (let i = 2; i <= N; i++) title.getCell(i).fill = HEADER_FILL;

  // 表头
  const hdr = s2.addRow(COLS.map((c) => c.h));
  hdr.height = 32;
  hdr.eachCell((c) => { c.font = F10W(); c.fill = HEADER_FILL; c.alignment = CENTER; c.border = BORDER; });

  // 数据行
  for (const rec of records) {
    const row = s2.addRow(rec.cells);
    row.height = 42;
    const gk = String(rec.cells[15]).replace(/[✅🟢🟡🟠🔴] /, "");
    const bg = (RATING_MAP[gk] || {}).fill || "FFFFFFFF";
    row.eachCell((cell, ci) => {
      cell.font = F10(false);
      cell.border = BORDER;
      cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: bg } };
      cell.alignment = [1, 2, 17].includes(ci) ? LEFT : CENTER;
    });
  }

  // 空行
  s2.addRow([]);

  // 体感评级图例标题
  const lt = s2.addRow(["📖 体感评级标准"]);
  lt.height = 26;
  lt.getCell(1).font = F11B();
  lt.getCell(1).fill = GRAY_FILL;
  lt.getCell(1).alignment = { vertical: "middle", wrapText: true };
  s2.mergeCells(lt.number, 1, lt.number, 4);
  for (let i = 2; i <= 4; i++) lt.getCell(i).fill = GRAY_FILL;

  // 图例表头
  const lh = s2.addRow(["等级", "阈值", "", "说明"]);
  lh.height = 22;
  [1, 2, 3, 4].forEach((i) => { lh.getCell(i).font = F10W(); lh.getCell(i).fill = HEADER_FILL; lh.getCell(i).alignment = CENTER; lh.getCell(i).border = BORDER; });
  s2.mergeCells(lh.number, 2, lh.number, 3);

  // 图例行
  for (const [label, rule, clr, desc] of RATING_ITEMS) {
    const lr = s2.addRow([label, rule, "", desc]);
    lr.height = 22;
    const f: ExcelJS.Fill = { type: "pattern", pattern: "solid", fgColor: { argb: `FF${clr}` } };
    lr.getCell(1).font = F10(true); lr.getCell(1).fill = f; lr.getCell(1).alignment = CENTER; lr.getCell(1).border = BORDER;
    lr.getCell(2).font = F10(true); lr.getCell(2).fill = f; lr.getCell(2).alignment = CENTER; lr.getCell(2).border = BORDER;
    lr.getCell(3).fill = f; lr.getCell(3).border = BORDER;
    lr.getCell(4).font = F10(false); lr.getCell(4).alignment = LEFT; lr.getCell(4).border = BORDER;
    s2.mergeCells(lr.number, 2, lr.number, 3);
  }

  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
