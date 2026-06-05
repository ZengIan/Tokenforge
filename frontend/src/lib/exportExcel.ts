import ExcelJS from "exceljs";
import type { EstimateResponse, GpuGroup, InferenceConfig, ModelSpec } from "../types";

export interface ExportRecord {
  id: string;
  label: string;
  cells: (string | number)[]; // 18 列, 顺序与模板一致
}

/* ================= 样式常量 (取自模板) ================= */
const FONT = "微软雅黑";
const TITLE_FILL = "FF2E5F8F"; // 标题行
const HEAD_FILL = "FF1F4E78";  // 表头
const BODY_FILL = "FFF7FBFE";  // 测算依据 说明列底色
const GRAY_FILL = "FFF2F2F2";  // 图例标题底色
const BORDER_RGB = "FFBFBFBF";

const border: Partial<ExcelJS.Borders> = {
  top: { style: "thin", color: { argb: BORDER_RGB } },
  bottom: { style: "thin", color: { argb: BORDER_RGB } },
  left: { style: "thin", color: { argb: BORDER_RGB } },
  right: { style: "thin", color: { argb: BORDER_RGB } },
};

const CENTER: Partial<ExcelJS.Alignment> = { vertical: "middle", horizontal: "center", wrapText: true };
const LEFT: Partial<ExcelJS.Alignment> = { vertical: "middle", horizontal: "left", wrapText: true };
const fill = (argb: string): ExcelJS.Fill => ({ type: "pattern", pattern: "solid", fgColor: { argb } });
const font = (o: Partial<ExcelJS.Font>): Partial<ExcelJS.Font> => ({ name: FONT, ...o });

/* ================= 体感评级 ================= */
interface Rating {
  label: string;
  fill: string;
  text: string;
  rule: string;
  desc: string;
}
const RATINGS: Rating[] = [
  { label: "✅ 流畅", fill: "FFD5F5DA", text: "FF267F2A", rule: "TPS ≥ 30 且 TTFT ≤ 2s", desc: "对话流畅，体感接近公共模型商用服务" },
  { label: "✅ 良好", fill: "FFE2F5DA", text: "FF267F2A", rule: "TPS ≥ 20 且 TTFT ≤ 5s", desc: "对话流畅，长输入完全可接受" },
  { label: "🟢 可接受", fill: "FFEAF7E6", text: "FF3B7A57", rule: "TPS ≥ 15 且 TTFT ≤ 10s", desc: "略慢但完全可用，适合内部生产" },
  { label: "🟡 偏慢", fill: "FFFFF4D6", text: "FFB07700", rule: "TPS ≥ 10 且 TTFT ≤ 20s", desc: "首字慢，吐字尚可，建议优化或限并发" },
  { label: "🟠 仅演示", fill: "FFFFE8D1", text: "FFC25A00", rule: "TPS ≥ 5 或 TTFT ≤ 40s", desc: "生产慎用，仅适合演示/批处理" },
  { label: "🔴 不可用", fill: "FFFFD6D6", text: "FFC00000", rule: "其他", desc: "体感差，建议降配置或扩资源" },
];
const fillOfLabel = (label: string) => RATINGS.find((r) => r.label === label)?.fill;

function ratingOf(tps: number, ttftMs: number): Rating {
  const s = ttftMs / 1000;
  if (tps >= 30 && s <= 2) return RATINGS[0];
  if (tps >= 20 && s <= 5) return RATINGS[1];
  if (tps >= 15 && s <= 10) return RATINGS[2];
  if (tps >= 10 && s <= 20) return RATINGS[3];
  if (tps >= 5 || s <= 40) return RATINGS[4];
  return RATINGS[5];
}

const n2 = (v: number) => Math.round(v * 100) / 100;

/* ================= 性能估算 18 列 (名称/列宽完全对齐模板) ================= */
const COLS: { h: string; w: number }[] = [
  { h: "算力卡类型", w: 18.1 },
  { h: "模型名称", w: 24.0 },
  { h: "权重大小(GB)", w: 14.5 },
  { h: "参数量(B)", w: 13.4 },
  { h: "量化类型", w: 14.1 },
  { h: "GPU配置(卡)", w: 14.8 },
  { h: "上下文(k)", w: 12.0 },
  { h: "期望\n并发数（个）", w: 13.5 },
  { h: "KV cache (GB)", w: 15.5 },
  { h: "激活值(GB)", w: 14.5 },
  { h: "框架开销(GB)", w: 14.2 },
  { h: "总显存开销(GB)\n(权重+激活+KV+框架)", w: 22.0 },
  { h: "可容纳\n并发数(个)", w: 13.0 },
  { h: "单用户 TPS\n(tokens/s)", w: 15.0 },
  { h: "首字延迟\nTTFT(ms)", w: 13.0 },
  { h: "每字延迟\nTPOT (ms)", w: 16.0 },
  { h: "体感评级", w: 13.0 },
  { h: "推理参数", w: 35.0 },
];
const N = COLS.length; // 18
const TITLE = "模型推理性能矩阵 · 按上下文长度递增、并发递减";

export function buildRecord(
  model: ModelSpec,
  gpuGroups: GpuGroup[],
  inf: InferenceConfig,
  r: EstimateResponse,
): ExportRecord {
  const card = gpuGroups[0]?.spec.name ?? "";
  const nGpu = gpuGroups.reduce((s, g) => s + g.count, 0);
  const ctxK = Math.round(inf.max_model_len / 1024);
  const lo = Math.round(r.single_tps_low);
  const hi = Math.round(r.single_tps_high);
  const rating = ratingOf(r.single_tps_low, r.ttft_ms);

  // 动态生成推理参数说明（对应 ③ 推理参数面板的所有 vLLM 启动参数）
  const infArgs: string[] = [
    `--max-model-len ${inf.max_model_len}`,
    `--max-num-seqs ${inf.max_num_seqs}`,
    `--max-num-batched-tokens ${inf.max_num_batched_tokens}`,
    `--max-num-prompt-tokens ${inf.input_len}`,
    `--gpu-memory-utilization ${inf.gpu_memory_utilization}`,
    `--dtype ${inf.dtype}`,
    `--quantization ${inf.quantization}`,
    `--kv-cache-dtype ${inf.kv_cache_dtype}`,
  ];
  if (inf.enforce_eager) infArgs.push("--enforce-eager");
  if (inf.async_scheduling) infArgs.push("--async-scheduling");
  if (inf.parallel_enabled) {
    infArgs.push(`--tensor-parallel-size ${inf.tp_size}`);
    infArgs.push(`--pipeline-parallel-size ${inf.pp_size}`);
    infArgs.push(`--data-parallel-size ${inf.dp_size}`);
  }

  const cells: (string | number)[] = [
    card,
    model.model_id,
    model.weight_size_gb!,
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
    lo === hi ? `${lo}` : `${lo}-${hi}`,
    Math.round(r.ttft_ms),
    n2(r.tpot_ms),
    rating.label,
    infArgs.map((a) => `  ${a}`).join("\n"),
  ];
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    label: `${card} ×${nGpu} · ${model.model_id} · ${ctxK}K/${inf.max_num_seqs}路`,
    cells,
  };
}

/* ================= 测算依据数据 ================= */
const JISUAN_HEADER = ["指标", "说明", "示例： Ascend 910C (128G) × 4 卡，DeepSeek-V4-Flash-w8a8-mtp (300.1B 参数，w8a8 量化)上下文：200K，期望并发：8"];
const JISUAN_ROWS: [string, string, string][] = [
  ["权重大小", "运行时权重显存 = 参数量(B) × 每参数字节 × 组量化开销\n每参数字节：\n• BF16/FP16 = 2 字节  • FP8/W8A8 = 1 字节\n• AWQ/GPTQ(4bit) = 0.5 字节\nMoE 模型：所有专家常驻显存，按总参数量计算", "300.1B × 1 字节 (w8a8) = 300.1 GB ≈ 300 GB"],
  ["KV Cache", "总占用 = 2 × 层数 × KV维度 × 上下文长度 × 并发数 × KV字节 ÷ 0.9（分页效率）\nKV维度 = KV头数 × head_dim（GQA架构）\n• 线性/混合注意力：按 kv_cache_factor 缩减（如 DeepSeek MLA 可压缩至 1/4）\n• 显示值为满并发、满上下文的真实需求", "KV 维度 = 1 (GQA头) × 512 (head_dim) = 512\n标准公式：2 × 43层 × 512 × 204800 × 8 × 1字节 ÷ 0.9\n         ≈ 783 GB  (若按标准 MHA)\nMLA 压缩系数约 0.205 ≈ 160.32 GB"],
  ["激活值", "max-num-batched-tokens × hidden_size × 精度字节 × 2\n（峰值中间张量经验系数，推理过程临时存储）", "≈ max-num-batched-tokens × hidden_size × 精度字节 × 2\n≈ 2048 × 4096 × 2 × 2\n≈ 0.032 GB"],
  ["框架开销", "1.2 GB/卡 × 卡数\n包含：\n• CUDA 上下文  • 算子 workspace（FlashAttention/MatMul 临时空间）\n• TP/PP 通信缓冲（AllReduce buffer）\n• vLLM/PyTorch 框架元数据\n开启 enforce-eager 每卡省约 0.6 GB\n注：卡数越多，固定开销越大", "标准值：1.2 GB/卡 × 4 = 4.8 GB"],
  ["总显存", "权重 + KV Cache + 激活值 + 框架开销\n判定标准：总显存 ≤ 卡数 × 单卡显存 × gpu-memory-utilization（默认 90%）\n若超限，vLLM 会自动降低并发（见\"可容纳并发\"）", "300.1 + 160.32 + 0.03 + 4.80 = 465.25 GB"],
  ["可容纳并发", "(显存预算 − 权重 − 激活值 − 框架开销) ÷ 每路 KV 显存\n显存预算 = 卡数 × 单卡显存 × gpu-memory-utilization", "每路 KV = 160.32 ÷ 8 = 20.04 GB\n可用 KV 空间 = 461 - 300.1 - 0.03 - 4.8 = 156.07 GB\n可容纳 = 156.07 ÷ 20.04 ≈ 7.8 → 取整 7 路"],
  ["单请求 TPS", "保守区间 = 1 ÷ 单 token 耗时\n单 token 耗时 = 激活权重读取时间 + 每 token 固定开销\n• 权重读取 = 激活权重字节 ÷ (聚合带宽 × 带宽利用率)\n• 固定开销 = 层数 × 每层开销 × 系数（MoE/线性/TP/eager）\n区间已按真实部署数据标定", "保守下限 = 1 ÷ (9.15 + 固定开销惩罚) ≈ 71\n上限 = 1 ÷ 9.15 × batch 收益 ≈ 140"],
  ["TTFT", "TTFT = 单请求prefill × 并发争抢(并发+1)/2 + 固定开销\n单请求prefill = (2×激活参数×输入长度 + 注意力O(n²)) ÷ (算力×利用率)\nprefill 是算力瓶颈：并发越高、prompt 越长，TTFT 越大", "并发1/2K≈几十ms\n并发8/2K≈数百ms\n并发8/32K≈数秒(此为平均, P99更高)"],
  ["TPOT", "TPOT = (激活权重 + 本路 KV) ÷ (带宽 × 利用率) + 固定开销\nKV 读取随上下文增长 → 长上下文 TPOT 变大", "激活权重 ≈ 10.99B × 1字节 = 10.99 GB\n本路 KV / 并发 ≈ 160GB / 8 = 20 GB/路\n(10.99 + 20) ÷ (3200 × 80%) ≈ 12 ms\n+ 固定开销 ≈ 4-5 ms → TPOT ≈ 16-17 ms"],
  ["跨机互联损耗", ">8 卡视为多机部署：\n• 高速无损（NVLink Switch/HCCS）：≈ 无损耗\n• InfiniBand/RoCE：约 -10%\n• 普通以太网：约 -25%，且每 token 跨机延迟更高\n单机内无 NVLink（纯 PCIe）：约 -15%", ""],
  ["量化类型", "• fp8/w8a8 = 8 bit（1 字节/参数）\n• awq/gptq/w4a8/w4a16 = 4 bit（0.5 字节/参数）\n• none = 按 dtype（BF16/FP16）\n量化越激进越省显存，但精度损失越大", ""],
  ["体感评级阈值", "✅ 流畅：TPS ≥ 30 且 TTFT ≤ 2s\n✅ 良好：TPS ≥ 20 且 TTFT ≤ 5s\n🟢 可接受：TPS ≥ 15 且 TTFT ≤ 10s\n🟡 偏慢：TPS ≥ 10 且 TTFT ≤ 20s\n🟠 仅演示:TPS ≥ 5 或 TTFT ≤ 40s\n🔴 离线/批处理：其他（TTFT 过长，仅适合离线任务）", ""],
  ["说明", "估算为理论近似值，实际推理性能以实测为准；\nsource=estimate 的卡型算力/带宽为占位值，性能仅供参考", ""]
];

/* ================= 1. 测算依据 Sheet (回退至第一个工作表) ================= */
function buildJisuanSheet(wb: ExcelJS.Workbook) {
  const ws = wb.addWorksheet("测算依据");
  ws.views = [{ showGridLines: true }]; // 强制显示网格线
  
  ws.getColumn(1).width = 18;
  ws.getColumn(2).width = 70.0; // 🔍 依据要求：说明列由 85 调紧凑至 70
  ws.getColumn(3).width = 60.0;

  // 表头自适应高度
  const h = ws.addRow(JISUAN_HEADER);
  const hLines = Math.max(...JISUAN_HEADER.map(val => String(val).split('\n').length));
  h.height = Math.max(hLines * 16 + 16, 42); 

  h.getCell(1).font = font({ bold: true, size: 11, color: { argb: "FFFFFFFF" } });
  h.getCell(2).font = font({ bold: true, size: 11, color: { argb: "FFFFFFFF" } });
  h.getCell(3).font = font({ bold: true, size: 10, color: { argb: "FFFFFFFF" } });
  for (let i = 1; i <= 3; i++) {
    h.getCell(i).fill = fill(HEAD_FILL);
    h.getCell(i).alignment = CENTER;
    h.getCell(i).border = border;
  }

  JISUAN_ROWS.forEach((row) => {
    const r = ws.addRow(row);
    
    // 基于动态折行数自适应计算最优行高，预防文本被裁剪
    const maxLines = Math.max(...row.map(val => String(val).split('\n').length));
    r.height = Math.max(maxLines * 16 + 16, 28); 

    r.getCell(1).font = font({ bold: true, size: 10 });
    r.getCell(1).alignment = CENTER;
    r.getCell(1).border = border;
    
    r.getCell(2).font = font({ size: 10 });
    r.getCell(2).alignment = LEFT;
    r.getCell(2).fill = fill(BODY_FILL);
    r.getCell(2).border = border;
    
    r.getCell(3).font = font({ size: 10 });
    r.getCell(3).alignment = LEFT;
    r.getCell(3).border = border;
  });
}

/* ================= 2. 性能估算 Sheet (作为第二个工作表) ================= */
function buildMatrixSheet(wb: ExcelJS.Workbook, records: ExportRecord[]) {
  const ws = wb.addWorksheet("性能估算");
  ws.views = [{ showGridLines: true }];
  
  COLS.forEach((c, i) => (ws.getColumn(i + 1).width = c.w));

  // 标题
  const title = ws.addRow([TITLE]);
  title.height = 36;
  ws.mergeCells(1, 1, 1, N);
  for (let i = 1; i <= N; i++) {
    const c = title.getCell(i);
    c.fill = fill(TITLE_FILL);
    c.font = font({ bold: true, size: 14, color: { argb: "FFFFFFFF" } });
    c.alignment = CENTER;
  }

  // 表头
  const hdr = ws.addRow(COLS.map((c) => c.h));
  hdr.height = 42;
  hdr.eachCell((c: ExcelJS.Cell, i: number) => {
    c.fill = fill(HEAD_FILL);
    c.font = font({ bold: true, size: i >= 3 && i <= 5 ? 11 : 10, color: { argb: "FFFFFFFF" } });
    c.alignment = CENTER;
    c.border = border;
  });

  // 数据行
  for (const rec of records) {
    const row = ws.addRow(rec.cells);
    row.height = 28;
    const ratingFill = fillOfLabel(String(rec.cells[16]));
    
    row.eachCell((c: ExcelJS.Cell, i: number) => {
      c.font = font({ size: 10 });
      c.border = border;
      c.alignment = i === 1 || i === 2 || i === 18 ? LEFT : CENTER;
      
      // 显存与开销等高频数字千分位及小数格式化
      if ([3, 4, 9, 10, 11, 12, 16].includes(i)) {
        c.numFmt = '#,##0.00';
      } else if ([6, 7, 8, 13, 15].includes(i)) {
        c.numFmt = '#,##0';
      }
      
      if (i === 17 && ratingFill) {
        c.fill = fill(ratingFill);
        const matched = RATINGS.find((r) => r.label === c.value);
        if (matched) c.font = font({ bold: true, size: 10, color: { argb: matched.text } });
      }
    });
  }

  // 图例区
  ws.addRow([]);
  const lt = ws.addRow([]);
  lt.height = 26;
  lt.getCell(2).value = "📖 体感评级标准";
  ws.mergeCells(lt.number, 2, lt.number, N);
  for (let i = 2; i <= N; i++) lt.getCell(i).fill = fill(GRAY_FILL);
  lt.getCell(2).font = font({ bold: true, size: 11 });
  lt.getCell(2).alignment = LEFT;

  for (const rt of RATINGS) {
    const lr = ws.addRow([]);
    lr.height = 24;
    lr.getCell(2).value = rt.label;
    lr.getCell(3).value = rt.rule;
    lr.getCell(6).value = rt.desc;
    
    ws.mergeCells(lr.number, 3, lr.number, 5);
    ws.mergeCells(lr.number, 6, lr.number, N);
    
    for (let i = 2; i <= N; i++) {
      const c = lr.getCell(i);
      c.border = border;
      if (i >= 2 && i <= 5) {
        c.fill = fill(rt.fill);
        c.font = font({ bold: true, size: 10, color: { argb: rt.text } });
        c.alignment = CENTER;
      } else if (i >= 6) {
        c.font = font({ size: 10 });
        c.alignment = LEFT;
      }
    }
  }
}

/* ================= 外部调用下载方法 ================= */
export async function downloadExcel(records: ExportRecord[], filename = "模型推理性能估算.xlsx") {
  const wb = new ExcelJS.Workbook();
  
  // 💡 调整顺序：将“测算依据”放到前面最左边，“性能估算”放到后面右边
  buildJisuanSheet(wb);
  buildMatrixSheet(wb, records);
  
  const buf = await wb.xlsx.writeBuffer();
  const blob = new Blob([buf], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
