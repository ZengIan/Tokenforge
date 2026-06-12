// 推理引擎 + 量化合法值的单一事实来源 (store 自动选择 与 InferencePanel 下拉 共用)
import type { Engine, GpuSpec, Quantization } from "../types";

/** 按 GPU 卡名匹配默认推理引擎 (PRD §1.2) */
export function engineForCard(name: string): Engine {
  const n = name.toLowerCase();
  if (
    name.includes("昇腾") || name.includes("华为") ||
    n.includes("ascend") || n.includes("huawei")
  ) {
    return "vllm-ascend";
  }
  if (
    name.includes("平头哥") ||
    n.includes("ppu") || n.includes("pg1")
  ) {
    return "sglang";
  }
  // NVIDIA / 海光 / 其他国产 fork 保守走标准 vLLM
  return "vllm";
}

/** 卡是否有原生 FP8 算力 (用 gpus.yaml 的 fp8_tflops, 比卡名硬匹配可靠) */
export function cardHasNativeFp8(spec?: GpuSpec | null): boolean {
  return !!spec && spec.fp8_tflops > 0;
}

/**
 * 引擎 × 卡能力 双重过滤的量化合法值 (PRD §2.4)。
 * fp8 仅在有原生 FP8 的卡上出现 (910C/PPU 的 fp8_tflops=0 → 自动隐藏)。
 */
export function quantOptionsFor(engine: Engine, hasFp8: boolean): Quantization[] {
  let opts: Quantization[];
  switch (engine) {
    case "vllm-ascend":
      opts = ["none", "ascend"];
      break;
    case "sglang":
      opts = ["none", "w8a8_int8", "awq", "gptq", "fp8"];
      break;
    case "vllm":
    default:
      // 无原生 FP8 的卡(A100/海光/沐曦/天数/昆仑/寒武纪等)给通用 INT8(w8a8)选项:
      // 选中即 1 字节权重 + 自动 2×FP16 算力(由后端按量化判别)。
      opts = hasFp8
        ? ["none", "fp8", "awq", "gptq", "bitsandbytes"]
        : ["none", "w8a8", "awq", "gptq", "bitsandbytes"];
      break;
  }
  return hasFp8 ? opts : opts.filter((q) => q !== "fp8");
}

/** 引擎推荐的默认量化 (切换引擎且当前值非法时回退到此) */
export function defaultQuantFor(engine: Engine): Quantization {
  if (engine === "vllm-ascend") return "ascend";
  if (engine === "sglang") return "w8a8_int8";
  return "none";
}

/**
 * 旧分享链接/记录里的废弃量化值 → 规范值映射 (hydrate 时清洗)。
 * w8a8 现在是标准 vLLM 的规范 INT8 值, 不再映射;
 * int8 归一到 w8a8; sglang/ascend 引擎下若不合法, coerceQuant 会再回退到引擎默认 INT8。
 */
const LEGACY_QUANT: Record<string, Quantization> = {
  int8: "w8a8",
  w4a8: "awq",
  w4a16: "awq",
};
export function mapLegacyQuant(q: string): Quantization {
  return (LEGACY_QUANT[q] ?? q) as Quantization;
}

/**
 * 给定引擎+卡, 把当前量化收敛到合法值:
 * 已合法则保留, 否则回退到引擎默认值。
 */
export function coerceQuant(
  engine: Engine,
  hasFp8: boolean,
  current: Quantization,
): Quantization {
  const mapped = mapLegacyQuant(current);
  const opts = quantOptionsFor(engine, hasFp8);
  return opts.includes(mapped) ? mapped : defaultQuantFor(engine);
}
