import { useEffect, useRef, useState } from "react";
import { fetchModelDetail, searchModels } from "../api";
import { useStore } from "../store";
import type { SearchResult } from "../types";

export function ModelPanel() {
  const LIMIT = 6;
  const { model, setModel } = useStore();
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
        <NumField label="参数量 (B)" value={model.params_b} onChange={(v) => setModel({ params_b: v })} decimals={2} />
        <NumField
          label="权重大小 (GB)"
          value={model.weight_size_gb ?? 0}
          onChange={(v) => setModel({ weight_size_gb: v })}
          decimals={2}
        />
        <NumField label="层数" value={model.num_layers} onChange={(v) => setModel({ num_layers: v })} />
        <NumField label="隐藏层" value={model.hidden_size} onChange={(v) => setModel({ hidden_size: v })} />
        <NumField label="注意力头数" value={model.num_attention_heads} onChange={(v) => setModel({ num_attention_heads: v })} />
        <NumField
          label="KV 头数 (GQA)"
          value={model.num_key_value_heads ?? model.num_attention_heads}
          onChange={(v) => setModel({ num_key_value_heads: v })}
        />
        <TextField label="默认模型精度" value={model.precision} onChange={(v) => setModel({ precision: v })} />
        <TextField label="张量类型" value={model.tensor_types ?? ""} onChange={(v) => setModel({ tensor_types: v } as any)} />
      </div>
      <p className="mt-2 text-[11px] text-slate-300">
        以上参数由 ModelScope 自动获取
        {(!model.tensor_types || model.tensor_types === "") && (
          <>，张量类型无法获取，请到{" "}
            <a
              href={`https://modelscope.cn/models/${model.model_id}`}
              target="_blank"
              rel="noreferrer"
              className="text-forge-ember underline hover:text-forge-flame"
            >
              ModelScope
            </a>
            {" "}确认
          </>
        )}。
      </p>
    </div>
  );
}

function NumField({ label, value, onChange, decimals }: { label: string; value: number; onChange: (v: number) => void; decimals?: number }) {
  return (
    <div>
      <span className="label">{label}</span>
      <input
        type="text"
        inputMode="decimal"
        className="input cursor-text"
        value={decimals != null ? value.toFixed(decimals) : value}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (!isNaN(n)) onChange(n);
        }}
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
