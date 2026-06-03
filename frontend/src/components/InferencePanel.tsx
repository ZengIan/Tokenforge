import { useStore } from "../store";
import type { DType, KVCacheDType, Quantization } from "../types";

const DTYPES: DType[] = ["auto", "float16", "bfloat16", "float32"];
const QUANTIZATIONS: Quantization[] = [
  "none",
  "fp8",
  "awq",
  "gptq",
  "int8",
  "w8a8",
  "w4a8",
  "w4a16",
];
const KV_DTYPES: KVCacheDType[] = ["auto", "fp8", "fp8_e5m2", "fp8_e4m3", "int8"];

const LEN_PRESETS = [
  { label: "4K", v: 4096 },
  { label: "8K", v: 8192 },
  { label: "32K", v: 32768 },
  { label: "128K", v: 131072 },
  { label: "256K", v: 262144 },
];

export function InferencePanel() {
  const { inference, setInference } = useStore();
  const i = inference;

  return (
    <div className="card">
      <h2 className="mb-1 text-sm font-bold text-forge-flame">③ 推理参数</h2>
      <p className="mb-3 text-[11px] text-slate-400">
        对应 vLLM 启动参数，默认即推荐配置；悬停 <Q /> 查看说明与调整影响。
      </p>

      <NumberField
        label="上下文长度"
        flag="--max-model-len"
        tip="单请求最大上下文（输入+输出）长度，KV Cache 按它每路预留。推荐：按业务最长需求设置（如 8K/32K）。调大 → 每路 KV 线性增大、可并发数下降；调小 → 省显存、并发更高。"
        value={i.max_model_len}
        min={512}
        max={1048576}
        onChange={(v) => setInference({ max_model_len: v })}
        presets={LEN_PRESETS}
        onPreset={(v) => setInference({ max_model_len: v })}
      />
      <NumberField
        label="并发数"
        flag="--max-num-seqs"
        tip="引擎同时处理的最大序列数（并发上限）。推荐：256。调大 → 总吞吐 TPS 上升，但 KV 显存占用上升；超过显存可容纳量时 vLLM 会自动排队。"
        value={i.max_num_seqs}
        min={1}
        max={8192}
        onChange={(v) => setInference({ max_num_seqs: v })}
      />
      <NumberField
        label="最大批处理 Token 数"
        flag="--max-num-batched-tokens"
        tip="单次迭代最多处理的 token 数（prefill 分块预算）。推荐：≥ max-model-len，常用 8192。调大 → prefill 更快、TTFT 更低，但峰值激活显存上升；调小 → 省显存但首字变慢。"
        value={i.max_num_batched_tokens}
        min={256}
        max={1048576}
        onChange={(v) => setInference({ max_num_batched_tokens: v })}
      />
      <NumberField
        label="显存利用率"
        flag="--gpu-memory-utilization"
        tip="允许 vLLM 使用的单卡显存比例。推荐：0.9。调大 → 更多显存留给 KV、并发更高，但 >0.95 易 OOM；调小 → 更安全但并发下降。"
        value={i.gpu_memory_utilization}
        min={0.1}
        max={1}
        step={0.05}
        onChange={(v) => setInference({ gpu_memory_utilization: v })}
      />

      <div className="mt-1 grid grid-cols-2 gap-2">
        <Select
          label="计算精度"
          flag="--dtype"
          tip="权重/激活的计算精度。推荐：auto（通常 BF16）。float32 → 精度最高但显存翻倍、最慢；float16 / bfloat16 → 标准选择，二者显存相同。"
          value={i.dtype}
          options={DTYPES}
          onChange={(v) => setInference({ dtype: v as DType })}
        />
        <Select
          label="模型量化"
          flag="--quantization"
          tip="权重量化方法。推荐：none（原始精度）或 fp8（H 卡）。awq/gptq（4bit）→ 权重显存降至约 1/4，适合显存紧张；fp8 → 减半且 H 卡有加速。越激进精度损失越大。"
          value={i.quantization}
          options={QUANTIZATIONS}
          onChange={(v) => setInference({ quantization: v as Quantization })}
        />
        <Select
          label="KV Cache 精度"
          flag="--kv-cache-dtype"
          tip="KV 缓存存储精度。推荐：auto（跟随 dtype）。选 fp8 → KV 显存减半、可并发翻倍，精度损失通常很小；int8 类似。"
          value={i.kv_cache_dtype}
          options={KV_DTYPES}
          onChange={(v) => setInference({ kv_cache_dtype: v as KVCacheDType })}
        />
        <div className="flex items-end">
          <Toggle
            label="禁用 CUDA Graph"
            flag="--enforce-eager"
            tip="关闭 CUDA Graph。推荐：关闭此项（即保持 CUDA Graph 开启）。开启 → 省约 1–3GB 显存、启动快，但 decode 吞吐下降约 5–15%；仅在显存极紧张或调试时开启。"
            checked={i.enforce_eager}
            onChange={(v) => setInference({ enforce_eager: v })}
          />
        </div>
      </div>
    </div>
  );
}

/** 悬停说明的问号徽标 */
function Tip({ text }: { text: string }) {
  return (
    <span className="group relative ml-1 inline-flex align-middle">
      <span className="flex h-3.5 w-3.5 cursor-help items-center justify-center rounded-full border border-slate-500 text-[9px] leading-none text-slate-400 group-hover:border-forge-ember group-hover:text-forge-ember">
        ?
      </span>
      <span className="pointer-events-none absolute left-5 top-0 z-30 hidden w-64 rounded-md border border-slate-600 bg-slate-900 p-2 text-[11px] font-normal leading-snug text-slate-200 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}

function Q() {
  return (
    <span className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-slate-500 text-[9px] text-slate-400">
      ?
    </span>
  );
}

function LabelRow({ label, flag, tip }: { label: string; flag: string; tip: string }) {
  return (
    <span className="label mb-0 flex items-center">
      {label}
      <code className="ml-1.5 rounded bg-slate-900/70 px-1 text-[10px] text-slate-400">
        {flag}
      </code>
      <Tip text={tip} />
    </span>
  );
}

function NumberField({
  label,
  flag,
  tip,
  value,
  min,
  max,
  step,
  onChange,
  presets,
  onPreset,
}: {
  label: string;
  flag: string;
  tip: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  presets?: { label: string; v: number }[];
  onPreset?: (v: number) => void;
}) {
  return (
    <div className="mb-2">
      <div className="flex items-center justify-between">
        <LabelRow label={label} flag={flag} tip={tip} />
        <input
          type="number"
          className="input w-28 py-0.5 text-right"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(clamp(Number(e.target.value), min, max))}
        />
      </div>
      {presets && (
        <div className="mt-1 flex gap-1">
          {presets.map((p) => (
            <button
              key={p.label}
              className={
                "chip text-[10px] " +
                (value === p.v ? "border-forge-ember text-forge-ember" : "")
              }
              onClick={() => onPreset?.(p.v)}
            >
              {p.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Select({
  label,
  flag,
  tip,
  value,
  options,
  onChange,
}: {
  label: string;
  flag: string;
  tip: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <LabelRow label={label} flag={flag} tip={tip} />
      <select className="input mt-1" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function Toggle({
  label,
  flag,
  tip,
  checked,
  onChange,
}: {
  label: string;
  flag: string;
  tip: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2">
      <input
        type="checkbox"
        className="accent-forge-ember"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <LabelRow label={label} flag={flag} tip={tip} />
    </label>
  );
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}
