import type { EncodedTile, ExportFormat, SplitBox } from '../types/splitter';
import { formatBoxMeta } from '../utils/splitterUtils';

interface SlicesGalleryProps {
  boxes: SplitBox[];
  tiles: EncodedTile[];
  selectedIndex: number;
  format: ExportFormat;
  onSelectBox: (idx: number) => void;
  onInspectTile: (idx: number) => void;
  onDownloadTile: (idx: number) => void;
}

export function SlicesGallery({
  boxes,
  tiles,
  selectedIndex,
  format,
  onSelectBox,
  onInspectTile,
  onDownloadTile,
}: SlicesGalleryProps) {
  const tileByIndex = new Map<number, EncodedTile>();
  for (const t of tiles) {
    tileByIndex.set(t.index, t);
  }

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-slate-900">切片输出结果</h2>
        <p className="font-mono text-xs tabular-nums text-slate-500">
          {boxes.length > 0
            ? `${boxes.length} 个切片 · 由 OpenCV C++ 编码 (${format.toUpperCase()})`
            : '未检测到切片'}
        </p>
      </div>

      {boxes.length === 0 ? (
        <div className="rounded-lg border border-slate-200 bg-white p-8 text-center text-xs text-slate-500">
          当前灵敏度与最小区域占比下未匹配到前景区域，请尝试降低灵敏度或切换为「网格」模式。
        </div>
      ) : (
        <div
          data-testid="slices-grid"
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3"
        >
          {boxes.map((box, idx) => {
            const tile = tileByIndex.get(box.index);
            const isSelected = idx === selectedIndex;
            const outW = tile?.width ?? box.width;
            const outH = tile?.height ?? box.height;
            const sizeKb = tile ? (tile.byteSize / 1024).toFixed(1) : '--';

            return (
              <article
                key={box.index}
                className={`flex flex-col justify-between rounded-lg border bg-white transition-colors duration-150 ${
                  isSelected ? 'border-blue-600' : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <button
                  type="button"
                  onClick={() => onSelectBox(idx)}
                  aria-label={`选择切片 ${box.index}`}
                  className="checker-bg flex h-44 w-full items-center justify-center overflow-hidden rounded-t-lg border-b border-slate-100 p-3 text-left focus-visible:outline-2 focus-visible:outline-blue-600"
                >
                  {tile ? (
                    <img
                      src={tile.dataUrl}
                      alt={`切片 ${box.index} (${outW}×${outH})`}
                      referrerPolicy="no-referrer"
                      className="max-h-full max-w-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-slate-400">正在编码预览...</span>
                  )}
                </button>

                <div className="space-y-2 p-3.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-xs font-semibold tabular-nums text-slate-900">
                      #{String(box.index).padStart(2, '0')} · {outW}×{outH}
                    </span>
                    <span className="font-mono text-[11px] tabular-nums text-slate-500">
                      {box.strategy} · {sizeKb} KB
                    </span>
                  </div>

                  <p className="truncate font-mono text-[11px] tabular-nums text-slate-500">
                    {formatBoxMeta(box)}
                  </p>

                  <div className="flex items-center gap-2 pt-1">
                    <button
                      type="button"
                      onClick={() => onInspectTile(idx)}
                      aria-label={`放大检查切片 ${box.index}`}
                      className="flex-1 rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50"
                    >
                      放大检查
                    </button>
                    <button
                      type="button"
                      onClick={() => onDownloadTile(idx)}
                      aria-label={`下载切片 ${box.index}`}
                      className="flex-1 rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50"
                    >
                      保存
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
