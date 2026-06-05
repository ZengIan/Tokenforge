import { useEffect, useRef, useState } from "react";
import { fetchModelDetail, searchModels } from "../api";
import { useStore } from "../store";
import type { Quantization, SearchResult } from "../types";

// 模型名里的量化标识 -> 量化方式(优先级从上到下)
const QUANT_NAME_HINTS: [string, Quantization][] = [
  ["w4a8", "w4a8"],
  ["w8a8", "w8a8"],
  ["w4a16", "w4a16"],
  ["awq", "awq"],
  ["gptq", "gptq"],
  ["fp8", "fp8"],
  ["int8", "int8"],
  ["int4", "w4a16"],
];

function quantFromName(modelId: string): Quantization | null {
  const low = modelId.toLowerCase();
  for (const [needle, q] of QUANT_NAME_HINTS) if (low.includes(needle)) return q;
  return null;
}

export function ModelPanel() {
  const LIMIT = 6;
  const { model, setModel, setInference } = useStore();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const timer = useRef<number>();
  const reqId = useRef(0);
  const pageRef = useRef(1);
  const skipNext = useRef(false); // 选中模型后跳过一次自动检索

  const isErr = (res: SearchResult[]) => res.length === 1 && !!res[0].error;

  // 第一页检索（输入框防抖 / 按钮 / 回车）
  async function freshSearch(query: string) {
    const text = query.trim();
    if (!text) {
      setResults([]);
      setHasMore(false);
      return;
    }
    const id = ++reqId.current;
    setLoading(true);
    try {
      const res = await searchModels(text, 1, LIMIT);
      if (id !== reqId.current) return; // 只采用最后一次请求，避免乱序覆盖
      setResults(res);
      setOpen(true);
      pageRef.current = 1;
      setHasMore(!isErr(res) && res.length >= LIMIT);
    } catch {
      if (id === reqId.current) {
        setResults([]);
        setHasMore(false);
      }
    } finally {
      if (id === reqId.current) setLoading(false);
    }
  }

  // 下滑触底加载下一页，追加到列表
  async function loadMore() {
    if (loadingMore || !hasMore) return;
    const text = q.trim();
    if (!text) return;
    const next = pageRef.current + 1;
    setLoadingMore(true);
    try {
      const res = await searchModels(text, next, LIMIT);
      if (!isErr(res) && res.length > 0) {
        setResults((prev) => {
          const seen = new Set(prev.map((p) => p.model_id));
          return [...prev, ...res.filter((r) => !seen.has(r.model_id))];
        });
        pageRef.current = next;
      }
      setHasMore(!isErr(res) && res.length >= LIMIT);
    } catch {
      /* 加载更多失败：静默，下次滚动可重试 */
    } finally {
      setLoadingMore(false);
    }
  }

  function onListScroll(e: React.UIEvent<HTMLUListElement>) {
    const el = e.currentTarget;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 48) loadMore();
  }

  // 输入时自动检索（防抖 500ms）；按钮/回车可立即触发
  useEffect(() => {
    window.clearTimeout(timer.current);
    if (skipNext.current) {
      skipNext.current = false; // 选中模型导致的 q 变化，不触发检索
      return;
    }
    if (!q.trim()) {
      setResults([]);
      setHasMore(false);
      return;
    }
    timer.current = window.setTimeout(() => freshSearch(q), 500);
    return () => window.clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  function searchNow() {
    window.clearTimeout(timer.current);
    freshSearch(q);
  }

  async function pick(r: SearchResult) {
    skipNext.current = true; // 防止填入模型名后又自动检索一次
    setOpen(false);
    setQ(r.model_id);
    // 按模型名自动匹配量化方式(如 ...-w4a8 / -fp8 / -awq); 名字不带量化标识 → none
    setInference({ quantization: quantFromName(r.model_id) ?? "none" });
    // 官方仓库大小 (StorageSize) 来自搜索结果,作为权重大小的权威来源
    try {
      const detail = await fetchModelDetail(r.model_id);
      setModel({
        ...detail,
        weight_size_gb: r.weight_size_gb ?? detail.weight_size_gb,
      });
    } catch {
      // 详情API失败: 保留当前参数,更新名称和权重大小
      setModel({
        model_id: r.model_id,
        params_accurate: false,
        weight_size_gb: r.weight_size_gb,
      });
    }
  }

  return (
    <div className="card">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-bold text-forge-flame">
        ① 模型选择
        <span className="text-[11px] font-normal text-slate-400">（参数由 ModelScope 自动获取，如有误差，请手动修改）</span>
      </h2>

      <div className="relative">
        <div className="flex gap-2">
          <input
            className="input flex-1"
            placeholder="关键字检索，如 qwen3-8B（回车或点搜索）"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onFocus={() => results.length && setOpen(true)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                searchNow();
              }
            }}
          />
          <button
            className="btn shrink-0 px-3"
            onClick={searchNow}
            disabled={loading || !q.trim()}
          >
            {loading ? "搜索中…" : "🔍 搜索"}
          </button>
        </div>
        {open && results.length > 0 && (
          <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-slate-600 bg-slate-900 shadow-xl">
            <div className="px-3 pt-2 text-xs font-bold text-forge-flame">模型库</div>
            <ul className="max-h-72 overflow-auto py-1" onScroll={onListScroll}>
              {results.map((r) => (
                <li
                  key={r.model_id}
                  className="cursor-pointer px-3 py-2 hover:bg-slate-800"
                  onClick={() => pick(r)}
                >
                  {r.error ? (
                    <span className="text-sm text-amber-400">
                      检索失败，可手动输入参数：{r.model_id}
                    </span>
                  ) : (
                    <>
                      <div className="text-sm font-medium text-slate-100">
                        {r.chinese_name || r.model_id}
                      </div>
                      <div className="mt-0.5 text-xs text-slate-300">{r.model_id}</div>
                    </>
                  )}
                </li>
              ))}
              {loadingMore && (
                <li className="px-3 py-2 text-center text-xs text-slate-400">
                  加载更多…
                </li>
              )}
              {!hasMore && !isErr(results) && results.length > 0 && (
                <li className="px-3 py-1.5 text-center text-[11px] text-slate-500">
                  没有更多了
                </li>
              )}
            </ul>
            <a
              href={`https://modelscope.cn/models?name=${encodeURIComponent(q)}`}
              target="_blank"
              rel="noreferrer"
              className="block border-t border-slate-800 px-3 py-2 text-center text-xs text-slate-300 hover:text-forge-ember"
            >
              → 在 ModelScope 查看更多“{q}”相关模型
            </a>
          </div>
        )}
      </div>

      {/* model parameters */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <NumField label="参数量 (B)" value={model.params_b} onChange={(v) => setModel({ params_b: v })} />
        <NumField label="权重大小 (GB)" value={model.weight_size_gb ?? 0} onChange={(v) => setModel({ weight_size_gb: v })} />
        <NumField
          label="层数"
          tip={`Transformer 的层数。每层 = 自注意力 + FFN。\n层数越多 → 理解力越强，但推理延迟和显存线性增长。\n影响 KV Cache 总大小：需 × 层数。`}
          value={model.num_layers} onChange={(v) => setModel({ num_layers: v })} />
        <NumField
          label="隐藏层"
          tip={`每层的向量维度，决定模型的"宽度"。\n维度越大 → 表达能力越强，但 KV Cache / 激活显存平方级增长。\n需被注意力头数整除（每头维度 = hidden_size ÷ 头数）。`}
          value={model.hidden_size} onChange={(v) => setModel({ hidden_size: v })} />
        <NumField
          label="注意力头数"
          tip={`把每层的注意力计算拆成多少个头并排算。\n头数越多特征越丰富，每头维度 = hidden_size ÷ 头数。\n必须整除 hidden_size，且 ≥ KV 头数。`}
          value={model.num_attention_heads} onChange={(v) => setModel({ num_attention_heads: v })} />
        <NumField
          label="KV 头数 (GQA/MLA/MHA/MQA)"
          tip={`背景：注意力 = 拿着问题(Q)翻关键词(K)找答案(V)。\nK 和 V 要存起来复用 → 这就是 KV Cache。\n长上下文时 KV Cache 显存会爆炸（200K 可能占几百 GB）。\n解决办法——少存点 K 和 V：\n\nMHA（标准）：每个提问角度都存一套自己的 K/V，显存吃满。\n例：Llama 3-8B（32 个 Q 头 + 32 套 KV）\n\nGQA（分组共享）：几个 Q 头共用一套 KV，省显存。\n例：Llama 3-70B（64Q 仅 8KV → 显存压到 1/8）\n\nMQA（极致省）：所有提问角度共享 1 套 K/V，精打细算。\n例：Gemma、Falcon\n\nMLA（压扁了存）：不用存完整 KV，把 KV 压成极小的低秩表示再还原。\n例：DeepSeek V2/V3/R1，KV 可压缩至几十分之一\n  使用方法：KV 头数设为 1，勾选下方"线性/混合注意力"，\n  填写 kv_cache_factor 压缩比`}
          value={model.num_key_value_heads ?? 0}
          onChange={(v) => setModel({ num_key_value_heads: v })}
        />
        <TextField label="默认模型精度" value={model.precision} onChange={(v) => setModel({ precision: v })} />
        <TextField label="张量类型" value={model.tensor_types ?? ""} onChange={(v) => setModel({ tensor_types: v } as any)} />
        <label className="block">
          <span className="label">注意力类型 (KV 关键)</span>
          <select
            className="input"
            value={model.attn_type || ""}
            onChange={(e) => setModel({ attn_type: e.target.value } as any)}
          >
            <option value="">自动识别</option>
            <option value="MHA">MHA（全头KV）</option>
            <option value="GQA">GQA（分组）</option>
            <option value="MQA">MQA（单KV头）</option>
            <option value="MLA">MLA（DeepSeek/GLM 低秩）</option>
          </select>
        </label>
        {model.attn_type === "MLA" ? (
          <NumField
            label="MLA latent 维度"
            value={model.mla_kv_dim ?? 576}
            onChange={(v) => setModel({ mla_kv_dim: v })}
          />
        ) : (
          <div />
        )}
      </div>

      {(model.is_moe || model.attn_type || model.is_linear_attn) && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {model.is_moe && (
            <span className="chip border-violet-600/60 text-violet-300">
              MoE · 激活 {model.active_params_b ?? "?"}B / 共 {model.params_b}B（只读激活权重）
            </span>
          )}
          {model.attn_type === "MLA" && (
            <span className="chip border-emerald-600/60 text-emerald-300">
              MLA · 低秩 KV(latent {model.mla_kv_dim ?? 576}) · 比 MHA 小 10~100×
            </span>
          )}
          {model.attn_type && model.attn_type !== "MLA" && (
            <span className="chip border-slate-500/60 text-slate-300">{model.attn_type} 注意力</span>
          )}
          {model.is_linear_attn && (
            <span className="chip border-sky-600/60 text-sky-300">
              线性/混合注意力 · KV×{model.kv_cache_factor ?? 1}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function NumField({ label, value, onChange, tip }: { label: string; value: number; onChange: (v: number) => void; tip?: string }) {
  const [local, setLocal] = useState(String(value));
  useEffect(() => { setLocal(String(value)); }, [value]);

  function commit() {
    const n = Number(local);
    if (!isNaN(n)) onChange(n);
    else setLocal(String(value));
  }

  return (
    <div>
      <span className="label flex items-center gap-1">
        {label}
        {tip && (
          <span className="group relative inline-flex align-middle">
            <span className="flex h-3.5 w-3.5 cursor-help items-center justify-center rounded-full border border-slate-500 text-[9px] leading-none text-slate-400 group-hover:border-forge-ember group-hover:text-forge-ember">
              ?
            </span>
            <span className="pointer-events-none absolute bottom-full right-0 z-50 mb-1 whitespace-pre-wrap rounded-lg border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-[11px] leading-relaxed text-slate-200 opacity-0 shadow-xl transition-opacity group-hover:opacity-100" style={{ minWidth: "360px", maxWidth: "480px" }}>
              {tip}
            </span>
          </span>
        )}
      </span>
      <input
        type="text"
        inputMode="decimal"
        className="input cursor-text"
        value={local}
        onChange={(e) => setLocal(e.target.value.replace(/[^0-9.]/g, ""))}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
      />
    </div>
  );
}

function TextField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <span className="label">{label}</span>
      <input
        type="text"
        className="input cursor-text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}
