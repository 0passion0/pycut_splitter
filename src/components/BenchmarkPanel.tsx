import type { BenchmarkResponse } from '../types/splitter';
import { summarizeBenchmark } from '../utils/splitterUtils';

interface BenchmarkPanelProps {
  data: BenchmarkResponse | null;
  running: boolean;
  error: string | null;
  onRunBenchmark: () => void;
}

export function BenchmarkPanel({ data, running, error, onRunBenchmark }: BenchmarkPanelProps) {
  const summary = data ? summarizeBenchmark(data.results) : null;

  return (
    <section className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-semibold text-slate-900">
            Python 3.12 OpenCV + NumPy 性能与内存基准测试
          </h1>
          <p className="mt-0.5 text-xs text-slate-500">
            在当前主机上实测真实版面检测、SIMD 形态学膨胀与多线程 PNG 编码（覆盖 1080p 至 8K 33.2 MP）。
          </p>
        </div>

        <button
          type="button"
          onClick={onRunBenchmark}
          disabled={running}
          className="shrink-0 whitespace-nowrap rounded-md bg-blue-600 px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
        >
          {running ? '正在运行 1080p–8K 压测...' : '运行实时基准'}
        </button>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-medium text-rose-700">
          {error}
        </div>
      )}

      {data && summary && (
        <>
          <div className="grid grid-cols-2 divide-x divide-slate-200 rounded-lg border border-slate-200 bg-white sm:grid-cols-4">
            <div className="px-4 py-3">
              <div className="text-[11px] font-medium text-slate-500">运行环境栈</div>
              <div className="mt-0.5 font-mono text-xs font-semibold text-slate-900">
                Py 3.12 · cv2 {data.cv2Version} · np {data.numpyVersion}
              </div>
            </div>
            <div className="px-4 py-3">
              <div className="text-[11px] font-medium text-slate-500">平均吞吐量</div>
              <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-slate-900">
                {summary.avgThroughputMp} MP/s
              </div>
            </div>
            <div className="px-4 py-3">
              <div className="text-[11px] font-medium text-slate-500">8K 峰值缓冲 / 原图</div>
              <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-emerald-700">
                {(summary.largestPeakRatio * 100).toFixed(1)}%
              </div>
            </div>
            <div className="px-4 py-3">
              <div className="text-[11px] font-medium text-slate-500">检测准确率</div>
              <div className="mt-0.5 font-mono text-sm font-semibold text-emerald-700">
                {summary.allAccurate ? '100% 精确' : '存在偏差'}
              </div>
            </div>
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-200 bg-white">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold text-slate-600">
                  <th className="px-4 py-2.5">测试负载</th>
                  <th className="px-4 py-2.5 text-right">像素 (MP)</th>
                  <th className="px-4 py-2.5">模式</th>
                  <th className="px-4 py-2.5 text-right">切片数</th>
                  <th className="px-4 py-2.5 text-right">解析耗时</th>
                  <th className="px-4 py-2.5 text-right">全流程 + PNG</th>
                  <th className="px-4 py-2.5 text-right">吞吐量</th>
                  <th className="px-4 py-2.5 text-right">峰值缓冲</th>
                  <th className="px-4 py-2.5 text-right">原始图像</th>
                  <th className="px-4 py-2.5 text-right">缓冲 / 原图</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums text-slate-800">
                {data.results.map((row) => {
                  const bufPct = row.rawImageMb > 0 ? (row.peakTransientMb / row.rawImageMb) * 100 : 0;
                  return (
                    <tr key={row.label} className="hover:bg-slate-50/70">
                      <td className="px-4 py-2.5 font-sans font-medium text-slate-900">{row.label}</td>
                      <td className="px-4 py-2.5 text-right">{row.megapixels.toFixed(2)}</td>
                      <td className="px-4 py-2.5">{row.mode}</td>
                      <td className="px-4 py-2.5 text-right">
                        {row.slices}/{row.expectedSlices}
                      </td>
                      <td className="px-4 py-2.5 text-right font-semibold text-blue-700">
                        {row.analyzeMs.toFixed(1)} ms
                      </td>
                      <td className="px-4 py-2.5 text-right">{row.fullPipelineMs.toFixed(1)} ms</td>
                      <td className="px-4 py-2.5 text-right">{row.throughputMpPerSec.toFixed(1)} MP/s</td>
                      <td className="px-4 py-2.5 text-right">{row.peakTransientMb.toFixed(2)} MB</td>
                      <td className="px-4 py-2.5 text-right text-slate-500">{row.rawImageMb.toFixed(2)} MB</td>
                      <td className="px-4 py-2.5 text-right text-slate-600">{bufPct.toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!data && !running && !error && (
        <div className="rounded-lg border border-slate-200 bg-white p-8 text-center">
          <p className="text-xs text-slate-600">
            点击 <strong>运行实时基准</strong>，即可直接在当前主机上评测 Python 3.12 + OpenCV + NumPy 在 1080p、2.8K、4K 与 8K 图像下的耗时与内存表现。
          </p>
        </div>
      )}
    </section>
  );
}
