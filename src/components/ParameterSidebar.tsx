import { useRef, useState } from 'react';
import type { AnalyzeParams, SplitMode, UniformMode } from '../types/splitter';
import { generatePythonSnippet, validateUniformTarget } from '../utils/splitterUtils';

interface ParameterSidebarProps {
  params: AnalyzeParams;
  sourceName: string;
  sourceWidth: number;
  sourceHeight: number;
  onUpdateParams: (patch: Partial<AnalyzeParams>) => void;
  onUploadFile: (file: File) => void;
  onLoad4KStress: () => void;
}

const MODES: { key: SplitMode; label: string }[] = [
  { key: 'auto', label: '自动' },
  { key: 'xycut', label: 'XY-Cut' },
  { key: 'contour', label: '轮廓' },
  { key: 'grid', label: '网格' },
];

const UNIFORM_MODES: { key: UniformMode; label: string }[] = [
  { key: 'pad', label: '填充' },
  { key: 'resize', label: '缩放' },
  { key: 'crop', label: '裁剪' },
];

export function ParameterSidebar({
  params,
  sourceName,
  sourceWidth,
  sourceHeight,
  onUpdateParams,
  onUploadFile,
  onLoad4KStress,
}: ParameterSidebarProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [copyLabel, setCopyLabel] = useState('复制');

  const uniformValidation = validateUniformTarget(
    params.uniformSize,
    params.targetWidth,
    params.targetHeight
  );

  const pythonSnippet = generatePythonSnippet(params, sourceName);

  const handleCopySnippet = async () => {
    try {
      await navigator.clipboard.writeText(pythonSnippet);
      setCopyLabel('已复制');
      setTimeout(() => setCopyLabel('复制'), 1500);
    } catch {
      setCopyLabel('手动复制');
    }
  };

  return (
    <aside
      aria-label="切分参数配置"
      className="w-full shrink-0 border-b border-slate-200 bg-white lg:w-[340px] lg:border-r lg:border-b-0"
    >
      <div className="divide-y divide-slate-200">
        {/* 输入图像 */}
        <section className="p-5">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h2 className="text-xs font-semibold tracking-wide text-slate-900">输入图像</h2>
            <span className="font-mono text-xs tabular-nums text-slate-500">
              {sourceWidth > 0 ? `${sourceWidth}×${sourceHeight}` : '--'}
            </span>
          </div>
          <label
            htmlFor="fileInput"
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer?.files?.[0];
              if (f) onUploadFile(f);
            }}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-4 text-center transition-colors duration-150 ${
              dragOver
                ? 'border-blue-600 bg-blue-50/50'
                : 'border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100/70'
            }`}
          >
            <input
              ref={fileInputRef}
              id="fileInput"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="sr-only"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUploadFile(f);
              }}
            />
            <span className="text-xs font-semibold text-slate-800">上传或拖拽图片</span>
            <span className="mt-1 text-[11px] text-slate-500">PNG / JPG / WebP（Python OpenCV 处理）</span>
          </label>
          <div className="mt-2.5 flex items-center justify-between gap-2">
            <span className="max-w-[180px] truncate font-mono text-[11px] text-slate-500" title={sourceName}>
              {sourceName}
            </span>
            <button
              type="button"
              onClick={onLoad4KStress}
              className="shrink-0 whitespace-nowrap rounded border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 transition-colors hover:bg-slate-50"
            >
              加载 4K 压测图
            </button>
          </div>
        </section>

        {/* 检测模式 */}
        <section className="p-5">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-slate-900">检测模式</h2>
          <div className="grid grid-cols-4 gap-1 rounded-lg bg-slate-100 p-1" role="group" aria-label="检测模式">
            {MODES.map((m) => (
              <button
                key={m.key}
                type="button"
                onClick={() => onUpdateParams({ mode: m.key })}
                className={`whitespace-nowrap rounded-md py-1.5 text-xs font-medium transition-colors duration-150 ${
                  params.mode === m.key
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </section>

        {/* 算法滑块或均匀网格设置 */}
        {params.mode === 'grid' ? (
          <section className="p-5">
            <h2 className="mb-3 text-xs font-semibold tracking-wide text-slate-900">均匀网格配置</h2>
            <div className="grid grid-cols-3 gap-2.5">
              <label className="flex flex-col gap-1 text-[11px] font-medium text-slate-600">
                <span>行数</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={params.rows}
                  onChange={(e) =>
                    onUpdateParams({ rows: Math.max(1, Math.min(20, Number.parseInt(e.target.value, 10) || 1)) })
                  }
                  className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-xs tabular-nums text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-slate-600">
                <span>列数</span>
                <input
                  type="number"
                  min={1}
                  max={20}
                  value={params.cols}
                  onChange={(e) =>
                    onUpdateParams({ cols: Math.max(1, Math.min(20, Number.parseInt(e.target.value, 10) || 1)) })
                  }
                  className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-xs tabular-nums text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-slate-600">
                <span>间距 (px)</span>
                <input
                  type="number"
                  min={0}
                  max={200}
                  value={params.gutterPx}
                  onChange={(e) =>
                    onUpdateParams({ gutterPx: Math.max(0, Math.min(200, Number.parseInt(e.target.value, 10) || 0)) })
                  }
                  className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-xs tabular-nums text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </label>
            </div>
          </section>
        ) : (
          <section className="space-y-4 p-5">
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="sensitivityInput" className="text-xs font-medium text-slate-700">
                  分割灵敏度
                </label>
                <output className="font-mono text-xs tabular-nums text-slate-600">
                  {params.sensitivity.toFixed(2)}
                </output>
              </div>
              <input
                id="sensitivityInput"
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={params.sensitivity}
                onChange={(e) => onUpdateParams({ sensitivity: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="minRatioInput" className="text-xs font-medium text-slate-700">
                  最小区域占比
                </label>
                <output className="font-mono text-xs tabular-nums text-slate-600">
                  {params.minRegionRatio.toFixed(2)}
                </output>
              </div>
              <input
                id="minRatioInput"
                type="range"
                min={0.03}
                max={0.35}
                step={0.01}
                value={params.minRegionRatio}
                onChange={(e) => onUpdateParams({ minRegionRatio: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="bleedTrimInput" className="text-xs font-medium text-slate-700">
                  边缘去污染
                </label>
                <output className="font-mono text-xs tabular-nums text-slate-600">{params.bleedTrim} px</output>
              </div>
              <input
                id="bleedTrimInput"
                type="range"
                min={0}
                max={8}
                step={1}
                value={params.bleedTrim}
                onChange={(e) => onUpdateParams({ bleedTrim: Number.parseInt(e.target.value, 10) })}
                className="w-full accent-blue-600"
              />
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="contourGapInput" className="text-xs font-medium text-slate-700">
                  轮廓闭合间隙
                </label>
                <output className="font-mono text-xs tabular-nums text-slate-600">{params.contourGapPx} px</output>
              </div>
              <input
                id="contourGapInput"
                type="range"
                min={1}
                max={24}
                step={1}
                value={params.contourGapPx}
                onChange={(e) => onUpdateParams({ contourGapPx: Number.parseInt(e.target.value, 10) })}
                className="w-full accent-blue-600"
              />
            </div>
          </section>
        )}

        {/* 统一输出尺寸 */}
        <section className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-semibold tracking-wide text-slate-900">统一输出尺寸</h2>
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-700">
            <input
              type="checkbox"
              checked={params.uniformSize}
              onChange={(e) => onUpdateParams({ uniformSize: e.target.checked })}
              className="h-4 w-4 rounded border-slate-300 accent-blue-600"
            />
            <span>将所有切片统一为相同尺寸</span>
          </label>

          <div className={`mt-3 space-y-3 ${params.uniformSize ? '' : 'pointer-events-none opacity-45'}`}>
            <div className="grid grid-cols-3 gap-1 rounded-lg bg-slate-100 p-1" role="group" aria-label="尺寸统一模式">
              {UNIFORM_MODES.map((um) => (
                <button
                  key={um.key}
                  type="button"
                  disabled={!params.uniformSize}
                  onClick={() => onUpdateParams({ uniformMode: um.key })}
                  className={`whitespace-nowrap rounded-md py-1.5 text-xs font-medium transition-colors duration-150 ${
                    params.uniformMode === um.key
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {um.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              <label className="flex flex-col gap-1 text-[11px] font-medium text-slate-600">
                <span>目标宽度</span>
                <input
                  type="number"
                  min={1}
                  placeholder="自动最大"
                  disabled={!params.uniformSize}
                  value={params.targetWidth}
                  onChange={(e) => onUpdateParams({ targetWidth: e.target.value })}
                  className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-xs tabular-nums text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </label>
              <label className="flex flex-col gap-1 text-[11px] font-medium text-slate-600">
                <span>目标高度</span>
                <input
                  type="number"
                  min={1}
                  placeholder="自动最大"
                  disabled={!params.uniformSize}
                  value={params.targetHeight}
                  onChange={(e) => onUpdateParams({ targetHeight: e.target.value })}
                  className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-xs tabular-nums text-slate-900 focus:border-blue-600 focus:outline-none"
                />
              </label>
            </div>
            {!uniformValidation.valid && uniformValidation.error && (
              <p className="text-[11px] font-medium text-rose-600" role="alert">
                {uniformValidation.error}
              </p>
            )}
          </div>
        </section>

        {/* 实时 Python API 调用代码 */}
        <section className="p-5">
          <div className="mb-2.5 flex items-center justify-between gap-2">
            <h2 className="text-xs font-semibold tracking-wide text-slate-900">Python API 调用代码</h2>
            <button
              type="button"
              onClick={handleCopySnippet}
              className="rounded border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 transition-colors hover:bg-slate-50"
            >
              {copyLabel}
            </button>
          </div>
          <pre className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50 p-3 font-mono text-[11px] leading-relaxed text-slate-800">
            {pythonSnippet}
          </pre>
        </section>
      </div>
    </aside>
  );
}
