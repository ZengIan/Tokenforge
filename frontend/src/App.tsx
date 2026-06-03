import { useEffect, useRef } from "react";
import { estimate, fetchGpus } from "./api";
import { GpuPanel } from "./components/GpuPanel";
import { InferencePanel } from "./components/InferencePanel";
import { ModelPanel } from "./components/ModelPanel";
import { ResultPanel } from "./components/ResultPanel";
import { useStore } from "./store";

export default function App() {
  const {
    setGpuDb,
    hydrateFromUrl,
    serializeToUrl,
    model,
    gpuGroups,
    inference,
    setResult,
    setLoading,
    setError,
  } = useStore();
  const debounce = useRef<number>();

  // load GPU database once
  useEffect(() => {
    fetchGpus()
      .then((g) => {
        setGpuDb(g);
        hydrateFromUrl();
      })
      .catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // auto re-estimate on any config change (debounce 500ms — PRD §5.2)
  useEffect(() => {
    if (gpuGroups.length === 0) return;
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        setResult(await estimate({ model, gpus: gpuGroups, inference }));
      } catch (e) {
        setError(String(e));
        setResult(null);
      } finally {
        setLoading(false);
      }
    }, 500);
    return () => window.clearTimeout(debounce.current);
  }, [model, gpuGroups, inference, setResult, setLoading, setError]);

  function share() {
    const url = serializeToUrl();
    navigator.clipboard?.writeText(url);
    history.replaceState(null, "", url);
    alert("分享链接已复制到剪贴板");
  }

  return (
    <div className="min-h-screen bg-forge-iron text-slate-100">
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-forge-iron/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-lg font-black tracking-tight">
              🔥 Tokenforge
            </h1>
            <p className="text-[11px] text-white">
              异构 GPU 推理吞吐与显存估算器 · Forge tokens from silicon
            </p>
          </div>
          <button className="btn" onClick={share}>
            分享配置
          </button>
        </div>
      </header>

      {/* 2×2 四宫格：左上 模型 / 左下 GPU / 右上 性能测算 / 右下 推理参数 */}
      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-4 p-4 lg:grid-cols-2">
        <div className="space-y-4">
          <ModelPanel />
          <GpuPanel />
        </div>
        <div className="space-y-4">
          <ResultPanel />
          <InferencePanel />
        </div>
      </main>

      <footer className="mx-auto max-w-7xl px-4 py-6 text-center text-[11px] text-white">
        估算为理论近似，实际部署误差目标 &lt;15%。国产卡参数请以厂商规格书或实测为准。
      </footer>
    </div>
  );
}
