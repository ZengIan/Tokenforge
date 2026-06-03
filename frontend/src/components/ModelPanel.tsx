import { useEffect, useRef, useState } from "react";
import { fetchModelDetail, searchModels } from "../api";
import { useStore } from "../store";
import type { SearchResult } from "../types";

export function ModelPanel() {
  const { model, setModel } = useStore();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const timer = useRef<number>();

  useEffect(() => {
    if (!q.trim()) {
      setResults([]);
      return;
    }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setLoading(true);
      try {
        setResults(await searchModels(q));
        setOpen(true);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300); // 300ms debounce — PRD §2.1.1
    return () => window.clearTimeout(timer.current);
  }, [q]);

  async function pick(r: SearchResult) {
    setOpen(false);
    setQ(r.model_id);
    try {
      const detail = await fetchModelDetail(r.model_id);
      setModel(detail);
    } catch {
      // graceful fallback: keep name + inferred params, let user edit manually
      setModel({ model_id: r.model_id, params_b: r.params_b || model.params_b });
    }
  }

  return (
    <div className="card">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-bold text-forge-flame">
        ① 模型选择
      </h2>

      <div className="relative">
        <input
          className="input"
          placeholder="搜索 ModelScope 模型，如 Qwen2.5-7B…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onFocus={() => results.length && setOpen(true)}
        />
        {loading && (
          <span className="absolute right-3 top-2 text-xs text-slate-500">…</span>
        )}
        {open && results.length > 0 && (
          <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-slate-600 bg-slate-900 shadow-xl">
            <div className="px-3 pt-2 text-xs font-bold text-forge-flame">模型库</div>
            <ul className="max-h-72 overflow-auto py-1">
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
                      <div className="mt-0.5 text-xs text-slate-500">{r.model_id}</div>
                    </>
                  )}
                </li>
              ))}
            </ul>
            <a
              href={`https://modelscope.cn/models?name=${encodeURIComponent(q)}`}
              target="_blank"
              rel="noreferrer"
              className="block border-t border-slate-800 px-3 py-2 text-center text-xs text-slate-500 hover:text-forge-ember"
            >
              → 在 ModelScope 查看更多“{q}”相关模型
            </a>
          </div>
        )}
      </div>

      {/* model parameters — editable manual override (PRD §2.1.3) */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Field label="参数量 (B)" value={model.params_b} onChange={(v) => setModel({ params_b: v })} />
        <Field label="层数" value={model.num_layers} onChange={(v) => setModel({ num_layers: v })} />
        <Field label="hidden_size" value={model.hidden_size} onChange={(v) => setModel({ hidden_size: v })} />
        <Field label="注意力头数" value={model.num_attention_heads} onChange={(v) => setModel({ num_attention_heads: v })} />
        <Field label="KV 头数 (GQA)" value={model.num_key_value_heads ?? model.num_attention_heads} onChange={(v) => setModel({ num_key_value_heads: v })} />
        <Field label="词表大小" value={model.vocab_size} onChange={(v) => setModel({ vocab_size: v })} />
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        数据可手动修改。GQA 的 KV 头数 &lt; 注意力头数时显著降低 KV Cache。
      </p>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <input
        type="number"
        className="input"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
