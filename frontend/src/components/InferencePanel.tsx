import { useStore } from "../store";
import type { Framework, Quant } from "../types";

const QUANTS: Quant[] = ["FP16", "BF16", "FP8", "INT8", "INT4", "GPTQ", "AWQ"];
const FRAMEWORKS: Framework[] = ["vLLM", "TensorRT-LLM", "SGLang", "llama.cpp"];
const KV_QUANTS = ["FP16", "BF16", "FP8", "INT8"] as const;

export function InferencePanel() {
  const { inference, setInference } = useStore();
  const i = inference;

  return (
    <div className="card">
      <h2 className="mb-3 text-sm font-bold text-forge-flame">③ 推理参数</h2>

      <Slider
        label="输入长度 (prefill)"
        value={i.input_len}
        min={1}
        max={131072}
        onChange={(v) => setInference({ input_len: v })}
      />
      <Slider
        label="输出长度 (decode)"
        value={i.output_len}
        min={1}
        max={131072}
        onChange={(v) => setInference({ output_len: v })}
      />
      <Slider
        label="并发数"
        value={i.concurrency}
        min={1}
        max={512}
        onChange={(v) => setInference({ concurrency: v })}
      />

      <div className="mt-2 grid grid-cols-2 gap-2">
        <Select
          label="量化方式"
          value={i.quant}
          options={QUANTS}
          onChange={(v) => setInference({ quant: v as Quant })}
        />
        <Select
          label="KV Cache 精度"
          value={i.kv_quant}
          options={[...KV_QUANTS]}
          onChange={(v) => setInference({ kv_quant: v as (typeof KV_QUANTS)[number] })}
        />
        <Select
          label="推理框架"
          value={i.framework}
          options={FRAMEWORKS}
          onChange={(v) => setInference({ framework: v as Framework })}
        />
        <label className="block">
          <span className="label">上下文长度 (推导)</span>
          <input className="input opacity-60" disabled value={i.input_len + i.output_len} />
        </label>
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
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
          onChange={(e) => onChange(clamp(Number(e.target.value), min, max))}
        />
      </div>
      <input
        type="range"
        className="w-full accent-forge-ember"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
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
    </label>
  );
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}
