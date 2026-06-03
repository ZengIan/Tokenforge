import { useStore } from "../store";
import type { Framework, Quant } from "../types";

const QUANTS: Quant[] = [
  "FP16",
  "BF16",
  "FP8",
  "INT8",
  "INT4",
  "GPTQ",
  "AWQ",
  "W8A8",
  "W4A8",
  "W4A16",
];
const FRAMEWORKS: Framework[] = ["vLLM", "TensorRT-LLM", "SGLang", "llama.cpp"];
const KV_QUANTS = ["FP16", "BF16", "FP8", "INT8"] as const;

const K = 1024;

export function InferencePanel() {
  const { inference, setInference } = useStore();
  const i = inference;

  return (
    <div className="card">
      <h2 className="mb-3 text-sm font-bold text-forge-flame">③ 推理参数</h2>

      <Slider
        label="输入长度 (prefill)"
        hint="单请求 Prompt 长度，决定首字延迟 TTFT"
        value={i.input_len}
        min={1}
        max={262144}
        onChange={(v) => setInference({ input_len: v })}
      />
      <Slider
        label="输出长度 (decode)"
        hint="单请求平均生成 token 数，决定总延迟"
        value={i.output_len}
        min={1}
        max={262144}
        onChange={(v) => setInference({ output_len: v })}
      />
      <Slider
        label="上下文长度 (context)"
        hint="KV Cache 按此长度预留，范围 8K–256K，可随意配置"
        value={i.context_len}
        min={8 * K}
        max={256 * K}
        step={8 * K}
        onChange={(v) => setInference({ context_len: v })}
      />
      <Slider
        label="并发数"
        hint="同时在飞的请求数，越高总 TPS 越高（直到触及算力/显存上限）"
        value={i.concurrency}
        min={1}
        max={512}
        onChange={(v) => setInference({ concurrency: v })}
      />

      <div className="mt-2 grid grid-cols-2 gap-2">
        <Select
          label="量化方式"
          hint="权重/激活位宽，如 W4A8 = 权重4bit 激活8bit"
          value={i.quant}
          options={QUANTS}
          onChange={(v) => setInference({ quant: v as Quant })}
        />
        <Select
          label="KV Cache 精度"
          hint="KV 缓存存储精度，降低可省显存"
          value={i.kv_quant}
          options={[...KV_QUANTS]}
          onChange={(v) => setInference({ kv_quant: v as (typeof KV_QUANTS)[number] })}
        />
        <Select
          label="推理框架"
          hint="影响算力/带宽利用率系数"
          value={i.framework}
          options={FRAMEWORKS}
          onChange={(v) => setInference({ framework: v as Framework })}
        />
        <NumberField
          label="gpu-memory-utilization"
          hint="vLLM 可用显存上限占比，默认 0.9；越高可容纳越多 KV"
          value={i.gpu_memory_utilization}
          min={0.1}
          max={1}
          step={0.05}
          onChange={(v) =>
            setInference({ gpu_memory_utilization: clamp(v, 0.1, 1) })
          }
        />
      </div>

      <Toggle
        label="enforce-eager"
        hint="关闭 CUDA Graph：省一点显存，但 decode 略慢（默认关闭）"
        checked={i.enforce_eager}
        onChange={(v) => setInference({ enforce_eager: v })}
      />
    </div>
  );
}

function Slider({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between">
        <span className="label mb-0">{label}</span>
        <input
          type="number"
          className="input w-24 py-0.5 text-right"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(clamp(Number(e.target.value), min, max))}
        />
      </div>
      <input
        type="range"
        className="w-full accent-forge-ember"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="-mt-1 text-[10px] text-slate-500">{hint}</p>}
    </div>
  );
}

function Select({
  label,
  hint,
  value,
  options,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      {hint && <p className="mt-0.5 text-[10px] text-slate-500">{hint}</p>}
    </label>
  );
}

function NumberField({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <input
        type="number"
        className="input"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="mt-0.5 text-[10px] text-slate-500">{hint}</p>}
    </label>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="mt-2 flex cursor-pointer items-start gap-2">
      <input
        type="checkbox"
        className="mt-0.5 accent-forge-ember"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>
        <span className="text-sm text-slate-200">{label}</span>
        {hint && <p className="text-[10px] text-slate-500">{hint}</p>}
      </span>
    </label>
  );
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}
