import { useEffect, useRef, useState } from 'react';
import type { EncodedTile, SplitBox } from '../types/splitter';

interface LightboxModalProps {
  open: boolean;
  box: SplitBox | null;
  tile: EncodedTile | null;
  onClose: () => void;
  onDownload: () => void;
}

export function LightboxModal({ open, box, tile, onClose, onDownload }: LightboxModalProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const [zoom2x, setZoom2x] = useState(false);

  useEffect(() => {
    const dlg = dialogRef.current;
    if (!dlg) return;
    if (open && !dlg.open) {
      dlg.showModal();
    } else if (!open && dlg.open) {
      dlg.close();
    }
  }, [open]);

  if (!box || !tile) {
    return <dialog ref={dialogRef} className="hidden" />;
  }

  const [tt, tr, tb, tl] = box.trimmedEdges;

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="lightboxTitle"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === dialogRef.current) onClose();
      }}
      className="m-auto max-h-[90vh] w-[min(92vw,960px)] rounded-xl border border-slate-200 bg-white p-0 shadow-xl backdrop:bg-slate-900/55"
    >
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 px-5 py-3.5">
        <div>
          <h3 id="lightboxTitle" className="text-sm font-semibold text-slate-900">
            切片 #{String(box.index).padStart(2, '0')} 边缘像素检查
          </h3>
          <p className="mt-0.5 font-mono text-xs tabular-nums text-slate-500">
            输出 {tile.width}×{tile.height} · 起点 ({box.x}, {box.y}) · 四周修边 {tt}/{tr}/{tb}/{tl}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-700">
            <input
              type="checkbox"
              checked={zoom2x}
              onChange={(e) => setZoom2x(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 accent-blue-600"
            />
            <span>2× 像素放大</span>
          </label>
          <button
            type="button"
            onClick={onDownload}
            className="rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
          >
            保存切片
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭预览"
            className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800"
          >
            关闭
          </button>
        </div>
      </div>

      <div className="checker-bg max-h-[75vh] overflow-auto p-6">
        <img
          src={tile.dataUrl}
          alt={`切片 ${box.index} 1:1 像素检视`}
          referrerPolicy="no-referrer"
          style={{
            width: zoom2x ? `${tile.width * 2}px` : `${tile.width}px`,
            height: zoom2x ? `${tile.height * 2}px` : `${tile.height}px`,
            imageRendering: 'pixelated',
          }}
          className="mx-auto block max-w-none border border-slate-200 shadow-xs"
        />
      </div>
    </dialog>
  );
}
