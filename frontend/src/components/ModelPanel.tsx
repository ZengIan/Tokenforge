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
  const reqId = useRef(0);

  async function doSearch(query: string) {
    const text = query.trim();
    if (!text) {
      setResults([]);
      return;
    }
    const id = ++reqId.current;
    setLoading(true);
    try {
      const res = await searchModels(text);
      if (id === reqId.current) {
        // 只采用最后一次请求的结果，避免乱序覆盖
        setResults(res);
        setOpen(true);
      }
    } catch {
      if (id === reqId.current) setResults([]);
    } finally {
      if (id === reqId.current) setLoading(false);
    }
  }

  // 输入时自动检索（防抖 500ms）；按钮/回车可立即触发
  useEffect(() => {
    window.clearTimeout(timer.current);
    if (!q.trim()) {
      setResults([]);
      return;
    }
    timer.current = window.setTimeout(() => doSearch(q), 500);
    return () => window.clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  function searchNow() {
    window.clearTimeout(timer.current);
    doSearch(q);
  }

  async function pick(r: SearchResult) {
    setOpen(false);
    setQ(r.model_id);
    // 官方仓库大小 (StorageSize) 来自搜索结果,作为权重大小的权威来源
    try {
      const detail = await fetchModelDetail(r.model_id);
      setModel({
        ...detail,
        weight_size_gb: r.weight_size_gb ?? detail.weight_size_gb,
      });
    } catch {
      // graceful fallback: keep name + inferred params, let user edit manually
      setModel({
        model_id: r.model_id,
        params_b: r.params_b || model.params_b,
        weight_size_gb: r.weight_size_gb,
      });
    }
  }

  return (
    <div className="card">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-bold text-forge-flame">
        ① 模型选择
      </h2>

      <div className="relative">
        <div className="flex gap-2">
          <input
            className="input flex-1"
            placeholder="关键字检索，如 qwen3-coder（回车或点搜索）"
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
                      <div className="mt-0.5 text-xs text-slate-300">{r.model_id}</div>
                    </>
                  )}
                </li>
              ))}
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

      {/* model parameters — auto-fetched, read-only */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <Field label="参数量 (B)" value={model.params_b} />
        <Field
          label="权重大小 (GB)"
          value={
            model.weight_size_gb != null
              ? `${model.weight_size_gb}（官方）`
              : `${weightGb(model.params_b, model.precision)}（估算）`
          }
        />
        <Field label="层数" value={model.num_layers} />
        <Field label="hidden_size" value={model.hidden_size} />
        <Field label="注意力头数" value={model.num_attention_heads} />
        <Field label="KV 头数 (GQA)" value={model.num_key_value_heads ?? model.num_attention_heads} />
        <Field label="词表大小" value={model.vocab_size} />
        <Field label="模型精度" value={model.precision} />
      </div>
      <p className="mt-2 text-[11px] text-slate-300">
        以上参数由 ModelScope 自动获取，不可修改。权重大小优先取官方仓库文件体积，缺失时按参数量 × 精度字节数估算。
      </p>
    </div>
  );
}

// 模型精度 -> 每参数字节数（用于权重大小展示）
const PRECISION_BYTES: Record<string, number> = {
  FP32: 4,
  FP16: 2,
  BF16: 2,
  FP8: 1,
  INT8: 1,
  INT4: 0.5,
  GPTQ: 0.5,
  AWQ: 0.5,
};

function weightGb(paramsB: number, precision: string): number {
  const bytes = PRECISION_BYTES[precision] ?? 2;
  return Math.round(((paramsB * 1e9 * bytes) / 1024 ** 3) * 10) / 10;
}

function Field({ label, value }: { label: string; value: number | string }) {
  return (
    <div>
      <span className="label">{label}</span>
      <div className="input cursor-default bg-slate-900/40 text-slate-300">{value}</div>
    </div>
  );
}
