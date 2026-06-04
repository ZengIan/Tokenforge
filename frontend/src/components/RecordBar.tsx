import { useStore } from "../store";
import { downloadCsv } from "../lib/exportCsv";

export function RecordBar() {
  const { result, records, addRecord, removeRecord, clearRecords } = useStore();

  return (
    <div className="card flex flex-col gap-2 p-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <button
          className="btn px-3 py-1.5 text-xs disabled:opacity-50"
          disabled={!result}
          onClick={addRecord}
          title={result ? "把当前测算结果加入对比清单" : "请先得到测算结果"}
        >
          ➕ 加入清单
        </button>
        <button
          className="rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-200 hover:border-forge-ember disabled:opacity-40"
          disabled={records.length === 0}
          onClick={() => downloadCsv(records)}
        >
          ⬇ 导出 CSV（{records.length}）
        </button>
        {records.length > 0 && (
          <button
            className="ml-auto text-[11px] text-slate-400 hover:text-red-400"
            onClick={clearRecords}
          >
            清空
          </button>
        )}
      </div>

      {records.length > 0 && (
        <ul className="flex flex-col gap-1">
          {records.map((r, i) => (
            <li
              key={r.id}
              className="flex items-center gap-2 rounded-md bg-slate-900/50 px-2 py-1 text-[11px] text-slate-300"
            >
              <span className="w-5 shrink-0 text-slate-500">{i + 1}.</span>
              <span className="flex-1 truncate" title={r.label}>
                {r.label}
              </span>
              <span className="shrink-0 text-slate-400">
                {r.cells[13]} tok/s · {r.cells[16]}
              </span>
              <button
                className="shrink-0 px-1 text-slate-500 hover:text-red-400"
                onClick={() => removeRecord(r.id)}
                title="移除"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
      {records.length === 0 && (
        <p className="text-[11px] text-slate-500">
          算完一组点「加入清单」，换卡/换模型继续，最后导出 CSV（含测算依据）。
        </p>
      )}
    </div>
  );
}
