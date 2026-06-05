import { useEffect, useState } from "react";
import { useStore } from "../store";
import type { DType, InterNode, KVCacheDType, Quantization } from "../types";

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

const SEQ_PRESETS = [
  { label: "8", v: 8 },
  { label: "16", v: 16 },
  { label: "32", v: 32 },
];


export function InferencePanel() {
  const { inference, setInference, gpuGroups } = useStore();
  const i = inference;
  const nGpu = gpuGroups.reduce((s, g) => s + g.count, 0);

  return (
    <div className="card">
      <h2 className="mb-1 text-sm font-bold text-forge-flame">③ 推理参数</h2>
      <p className="mb-3 text-[11px] text-slate-400">
        对应 vLLM 启动参数，默认即推荐配置；悬停 <Q /> 查看说明与调整影响。
      </p>

      <NumberField
        label="上下文长度"
        flag="--max-model-len"
        tip={
          "作用：一次对话最多能处理多少字（token）——你的问题（输入）加模型的回答（输出）合起来的上限。设 8K 就是输入+输出总共最多约 8000 个 token。\n\n" +
          "推荐：普通问答 8K 够用；长文档、长代码可设 32K 以上。\n\n" +
          "调大：每个请求要预留更多显存来记住前文（这块缓存叫 KV Cache），结果是能同时服务的人数变少。\n" +
          "调小：省显存、能同时服务更多人，但超长的输入会被截断。"
        }
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
        tip={
          "作用：服务器最多同时处理多少个请求，也就是多少人能同时在用。\n\n" +
          "推荐：16。\n\n" +
          "调大：同一时间能服务更多人，整体产出更高（产出用 TPS 衡量，即每秒生成多少个字/token，越高越快越能扛）。但每个请求都要占显存，显存不够时多出来的请求会自动排队等待。\n" +
          "调小：显存压力小，但用的人一多就容易排队。"
        }
        value={i.max_num_seqs}
        min={1}
        max={8192}
        onChange={(v) => setInference({ max_num_seqs: v })}
        presets={SEQ_PRESETS}
        onPreset={(v) => setInference({ max_num_seqs: v })}
      />
      <NumberField
        label="最大批处理 Token 数"
        flag="--max-num-batched-tokens"
        tip={
          '作用：服务器"读题"阶段一次最多并行读多少个字。模型在开口回答前，要先把你的问题整段读一遍理解，这个阶段业内叫 prefill（预填充）。\n\n' +
          '默认 2048：在线服务、追求高并发、prompt 不算特别长——绝大多数生产场景就用默认，够好，别瞎调。\n' +
          '调大到 4096 / 8192：prompt 很长（RAG、长文档）、想压低首 token 延迟、或离线批处理追 prefill 吞吐。代价是吃显存、长 prompt 时可能短暂挤压 decode。\n' +
          '调小到 512 / 1024：显存紧张、或并发解码优先于 prefill 速度（想让 decode 更平滑、TTFT 可以牺牲）。'
        }
        value={i.max_num_batched_tokens}
        min={256}
        max={1048576}
        onChange={(v) => setInference({ max_num_batched_tokens: v })}
      />
      <NumberField
        label="输入长度 (Prompt)"
        flag="仅影响 TTFT"
        tip={
          "--max-num-prompt-tokens 你的问题/上下文有多少个字(token)——只用来估算首字延迟(TTFT)。\n\n" +
          "为什么单独给：TTFT 几乎完全取决于 prompt 长度。\n" +
          "• 短输入(几百~2K，普通对话)：多卡下 TTFT 就是几十~几百毫秒(正常)。\n" +
          "• 长输入(4K~32K+，RAG/长文档/多轮历史)：TTFT 会到 0.5~3 秒甚至更久。\n" +
          "• 超长(128K+)：注意力 O(n²) 让 TTFT 进一步飙升。\n\n" +
          "按你实际业务的典型 prompt 长度填，TTFT 才准。"
        }
        value={i.input_len}
        min={1}
        max={1048576}
        onChange={(v) => setInference({ input_len: v })}
        presets={[
          { label: "512", v: 512 },
          { label: "2K", v: 2048 },
          { label: "8K", v: 8192 },
          { label: "32K", v: 32768 },
        ]}
        onPreset={(v) => setInference({ input_len: v })}
      />
      <NumberField
        label="显存利用率"
        flag="--gpu-memory-utilization"
        tip={
          "作用：允许程序占用单张显卡多大比例的显存。0.9 就是最多用 90%，留 10% 给系统和临时开销。\n\n" +
          "推荐：0.9。\n\n" +
          '调大：留给"前文缓存"（KV Cache）的空间更多，能同时服务更多人。但超过 0.95 很容易爆显存——也就是 OOM（显存不够导致程序崩溃）。\n' +
          "调小：更安全不易崩，但能服务的人数变少。"
        }
        value={i.gpu_memory_utilization}
        min={0.10}
        max={1}
        hasDecimal
        onChange={(v) => setInference({ gpu_memory_utilization: v })}
      />

      <div className="mt-1 grid grid-cols-2 gap-2">
        <Select
          label="计算精度"
          flag="--dtype"
          tip={
            '作用：模型计算时用多"精细"的数字。越精细结果越准，但越占显存、越慢。\n\n' +
            "推荐：auto（自动，通常是 bfloat16）。\n\n" +
            'float16 / bfloat16：都是"半精度"，显存占用一样，是常规选择。\n' +
            'float32："全精度"，显存直接翻倍、速度最慢，一般不需要。'
          }
          value={i.dtype}
          options={DTYPES}
          onChange={(v) => setInference({ dtype: v as DType })}
        />
        <Select
          label="模型量化"
          flag="--quantization"
          tip={
            '作用：把模型"压缩"成更小的格式来省显存，类似把高清图压成更小的文件。\n\n' +
            "推荐：none（不压缩，质量最好）；用 H 系列显卡可选 fp8。\n\n" +
            "awq / gptq：压成 4bit，权重显存能降到约原来的 1/4，适合显存不够的情况。\n" +
            "fp8：压一半，且在新显卡（H 系列）上还能跑得更快。\n" +
            "压得越狠越省显存，但回答质量损失也越大。"
          }
          value={i.quantization}
          options={QUANTIZATIONS}
          onChange={(v) => setInference({ quantization: v as Quantization })}
        />
        <Select
          label="KV Cache 精度"
          flag="--kv-cache-dtype"
          tip={
            "作用：模型用来记住前文的缓存（KV Cache）用多精细的数字存。这块缓存往往是显存大户。\n\n" +
            "推荐：auto（跟随上面的计算精度）。\n\n" +
            "选 fp8（或 int8）：这块缓存占的显存减半，差不多等于能同时服务的人数翻倍，而且对回答质量影响通常很小，性价比很高。"
          }
          value={i.kv_cache_dtype}
          options={KV_DTYPES}
          onChange={(v) => setInference({ kv_cache_dtype: v as KVCacheDType })}
        />
        <div className="flex items-end">
          <Toggle
            label="禁用 CUDA Graph"
            flag="--enforce-eager"
            tip={
              '作用：CUDA Graph 是一种让显卡跑得更快的优化。这个开关是"把这个优化关掉"。\n\n' +
              "推荐：不要勾（也就是保持优化开启）。\n\n" +
              "勾选后：能省约 1–3GB 显存、启动更快；但逐字往外蹦回答的阶段（叫 decode，解码）会慢 5–15%。只在显存特别紧张、或排查问题时才勾。"
            }
            checked={i.enforce_eager}
            onChange={(v) => setInference({ enforce_eager: v })}
          />
        </div>
        <div className="flex items-end">
          <Toggle
            label="异步调度"
            flag="--async-scheduling"
            tip={
              "作用：让调度器异步运行，GPU 执行当前 batch 时 CPU 并行准备下一个 batch，实现计算与调度的流水线重叠。\n\n" +
              "推荐：高并发在线服务建议开启。\n\n" +
              "开启后：减少 GPU 等待调度的时间，总吞吐可提升 10–30%；TPOT 略降。显存几乎无额外开销。\n" +
              "低并发/单请求场景收益很小。"
            }
            checked={i.async_scheduling}
            onChange={(v) => setInference({ async_scheduling: v })}
          />
        </div>
      </div>

      {/* 跨机互联 (真多机时) */}
      {nGpu > i.gpus_per_node && (
        <div className="mt-2">
          <Select
            label={`跨机互联（${Math.ceil(nGpu / Math.max(1, i.gpus_per_node))} 机 ${nGpu} 卡）`}
            flag="多机部署"
            tip={
              "卡数超过单机卡数 = 多机部署，跨机通信比机内 NVLink/HCCS 慢，会拖低算力效率。\n\n" +
              "• 高速无损(NVLink Switch/HCCS)：跨机也是高速互联，几乎无损耗。\n" +
              "• InfiniBand/RoCE：常规高速网，约 10% 损耗。\n" +
              "• 普通以太网：约 25% 损耗，且每 token 跨机延迟更高。"
            }
            value={i.internode}
            options={["nvlink", "ib", "ethernet"]}
            optionLabels={{
              nvlink: "高速无损 (NVLink Switch/HCCS)",
              ib: "InfiniBand / RoCE 高速网",
              ethernet: "普通以太网",
            }}
            onChange={(v) => setInference({ internode: v as InterNode })}
          />
        </div>
      )}

      {/* 并行配置 (TP/PP/DP) */}
      <div className="mt-2">
        <Toggle
          label="自定义并行 (TP/PP/DP)"
          flag="--*-parallel-size"
          tip={
            "是否手动指定并行切分。不勾选时默认按总卡数做张量并行（TP=卡数，PP=DP=1）。\n\n" +
            "• --tensor-parallel-size TP张量并行：把每一层横切到多卡并行算，降单卡显存与单请求延迟。\n" +
            "• --pipeline-parallel-size PP流水线并行：把不同层分到不同卡(流水线)，省卡间带宽、适合跨机，但有流水 bubble。\n" +
            "• --data-parallel-size DP数据并行：整套模型复制多份各服务不同请求，提升总吞吐(每份都占一份权重显存)。\n\n" +
            "⚠ 三者乘积必须 = 总卡数；TP 必须整除注意力头数；PP 必须整除层数，否则模型直接起不来。"
          }
          checked={i.parallel_enabled}
          onChange={(v) =>
            setInference(
              v
                ? { parallel_enabled: true, tp_size: nGpu, pp_size: 1, dp_size: 1 }
                : { parallel_enabled: false },
            )
          }
        />
        {i.parallel_enabled && (
          <div className="mt-1 grid grid-cols-3 gap-2">
            <ParField
              label="张量并行 TP"
              flag="TP"
              tip={
                "把模型每一层横切到 TP 张卡上并行算。\n• 降单卡权重/激活显存、降单请求延迟。\n• 需 TP 整除注意力头数(及 KV 头数)。\n• TP 越大卡间 all-reduce 越多，建议不超过单机卡数。"
              }
              value={i.tp_size}
              min={1}
              max={1024}
              onChange={(v) => setInference({ tp_size: v })}
            />
            <ParField
              label="流水线 PP"
              flag="PP"
              tip={
                "把模型按层数纵向切成 PP 段分到不同卡(流水线)。\n• 卡间只传激活、省带宽，适合跨机。\n• 需 PP 整除层数；有流水线 bubble，单请求延迟略增。\n• 与 --max-num-batched-tokens 相关：批太小流水线填不满、吞吐下降。"
              }
              value={i.pp_size}
              min={1}
              max={256}
              onChange={(v) => setInference({ pp_size: v })}
            />
            <ParField
              label="数据并行 DP"
              flag="DP"
              tip={
                "整套模型复制 DP 份，各自服务不同请求。\n• 总吞吐近似 ×DP，卡间几乎无通信。\n• 每份都要完整放下权重→单卡权重显存不随 DP 降低。\n• 适合显存放得下、想堆吞吐的场景。"
              }
              value={i.dp_size}
              min={1}
              max={256}
              onChange={(v) => setInference({ dp_size: v })}
            />
          </div>
        )}
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
      <span className="pointer-events-none absolute bottom-full right-0 z-30 mb-1 hidden w-80 whitespace-pre-line rounded-md border border-slate-600 bg-slate-900 p-3 text-[11px] font-normal leading-relaxed text-slate-200 shadow-xl group-hover:block">
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
      <code className="ml-1.5 shrink-0 rounded bg-slate-900/70 px-1 text-[10px] text-slate-400">
        {flag}
      </code>
      <Tip text={tip} />
    </span>
  );
}

/** 并行配置专用：标签在上，输入框在下 */
function ParField({
  label, flag, tip, value, min, max, onChange,
}: {
  label: string; flag: string; tip: string;
  value: number; min: number; max: number;
  onChange: (v: number) => void;
}) {
  const [local, setLocal] = useState(String(value));
  useEffect(() => { setLocal(String(value)); }, [value]);

  function commit() {
    const n = Number(local);
    if (!isNaN(n)) onChange(clamp(n, min, max));
    else setLocal(String(value));
  }

  return (
    <div className="flex flex-col gap-1">
      <LabelRow label={label} flag={flag} tip={tip} />
      <input
        type="text"
        inputMode="numeric"
        className="input w-full cursor-text"
        value={local}
        onChange={(e) => setLocal(e.target.value.replace(/\D/g, ""))}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
      />
    </div>
  );
}

function NumberField({
  label,
  flag,
  tip,
  value,
  min,
  max,
  onChange,
  presets,
  onPreset,
  hasDecimal,
}: {
  label: string;
  flag: string;
  tip: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  presets?: { label: string; v: number }[];
  onPreset?: (v: number) => void;
  hasDecimal?: boolean;
}) {
  const allowDot = hasDecimal === true;
  const [local, setLocal] = useState(String(value));
  useEffect(() => { setLocal(String(value)); }, [value]);

  function commit() {
    const n = Number(local);
    if (!isNaN(n)) onChange(clamp(n, min, max));
    else setLocal(String(value));
  }

  return (
    <div className="mb-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <LabelRow label={label} flag={flag} tip={tip} />
          {presets && (
            <div className="flex gap-1">
              {presets.map((p) => (
                <button
                  key={p.label}
                  className={
                    "chip text-[10px] " +
                    (value === p.v ? "border-forge-ember text-forge-ember" : "")
                  }
                  onClick={() => { onPreset?.(p.v); setLocal(String(p.v)); }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>
        <input
          type="text"
          inputMode={allowDot ? "decimal" : "numeric"}
          className="input w-32 shrink-0 cursor-text text-right"
          value={local}
          onChange={(e) => setLocal(e.target.value.replace(allowDot ? /[^0-9.]/g : /\D/g, ""))}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
      </div>
    </div>
  );
}

function Select({
  label,
  flag,
  tip,
  value,
  options,
  optionLabels,
  onChange,
}: {
  label: string;
  flag: string;
  tip: string;
  value: string;
  options: string[];
  optionLabels?: Record<string, string>;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <LabelRow label={label} flag={flag} tip={tip} />
      <select className="input mt-1" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o} value={o}>
            {optionLabels?.[o] ?? o}
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
