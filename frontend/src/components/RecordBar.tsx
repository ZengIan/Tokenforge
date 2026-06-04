import { useStore } from "../store";
import { downloadExcel } from "../lib/exportExcel";

export function RecordBar() {
  const { result, records, addRecord, clearRecords } = useStore();

  return (
    <div className="flex items-center gap-2">
      <button
        className="btn px-3 py-1.5 text-xs disabled:opacity-50"
        disabled={!result}
        onClick={addRecord}
        title={result ? "把当前测算结果加入对比清单" : "请先得到测算结果"}
      >
        ➕ 加入清单
      </button>
      {records.length > 0 && (
        <>
          <button
            className="rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-200 hover:border-forge-ember"
            onClick={() => downloadExcel(records)}
          >
            ⬇ 导出 Excel（{records.length}）
          </button>
          <button
            className="text-[11px] text-slate-400 hover:text-red-400"
            onClick={clearRecords}
          >
            清空
          </button>
        </>
      )}
    </div>
  );
}
