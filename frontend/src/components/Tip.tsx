/** 通用悬浮说明问号徽标。父容器需 relative,把它放到右上角即可。 */
export function Tip({ text, className = "" }: { text: string; className?: string }) {
  return (
    <span className={"group/tip relative inline-flex " + className}>
      <span className="flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-slate-500 text-[10px] leading-none text-slate-400 hover:border-forge-ember hover:text-forge-ember">
        ?
      </span>
      <span className="pointer-events-none absolute right-0 top-6 z-30 hidden w-72 whitespace-pre-line rounded-md border border-slate-600 bg-slate-900 p-3 text-left text-[11px] font-normal leading-relaxed text-slate-200 shadow-xl group-hover/tip:block">
        {text}
      </span>
    </span>
  );
}
