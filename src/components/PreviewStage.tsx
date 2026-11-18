import { useEffect, useRef, useState } from 'react';
import type { AnalyzeResponse, ExportFormat, SplitBox } from '../types/splitter';
import { formatBoxMeta } from '../utils/splitterUtils';

interface PreviewStageProps {
  sourceImageUrl: string | null;
  analysis: AnalyzeResponse | null;
  loading: boolean;
  error: string | null;
  selectedIndex: number;
  showOverlay: boolean;
  showSeamEnergy: boolean;
  format: ExportFormat;
  onToggleOverlay: (next: boolean) => void;
  onToggleSeamEnergy: (next: boolean) => void;
  onChangeFormat: (fmt: ExportFormat) => void;
  onSelectBox: (idx: number) => void;
}

export function PreviewStage({
  sourceImageUrl,
  analysis,
  loading,
  error,
  selectedIndex,
  showOverlay,
  showSeamEnergy,
  format,
  onToggleOverlay,
  onToggleSeamEnergy,
  onChangeFormat,
  onSelectBox,
}: PreviewStageProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [loadedSrc, setLoadedSrc] = useState<string | null>(null);
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number } | null>(null);

  // Load source image into HTMLImageElement whenever URL changes
  useEffect(() => {
    if (!sourceImageUrl) {
      imgRef.current = null;
      setLoadedSrc(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled) {
        imgRef.current = img;
        setLoadedSrc(sourceImageUrl);
      }
    };
    img.src = sourceImageUrl;
    return () => {
      cancelled = true;
    };
  }, [sourceImageUrl]);

  // Draw source image + split box overlays + optional NumPy seam energy curves
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !analysis) return;

    const w = analysis.imageWidth || img.naturalWidth;
    const h = analysis.imageHeight || img.naturalHeight;
    if (w <= 0 || h <= 0) return;

    // Fit crisp preview canvas resolution up to 1600px wide for smooth 60fps canvas redraws
    const scale = w > 1600 ? 1600 / w : 1;
    const cw = Math.round(w * scale);
    const ch = Math.round(h * scale);

    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(img, 0, 0, cw, ch);

    // Optional NumPy Seam Energy Curves overlay along top & left margins (drawn underneath labels)
    if (showSeamEnergy && analysis.diagnostics?.seam_curves) {
      const { h: hCurve, v: vCurve } = analysis.diagnostics.seam_curves;
      ctx.save();
      if (vCurve && vCurve.length > 1) {
        const maxV = Math.max(1, ...vCurve);
        const bandH = Math.min(46, ch * 0.075);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.48)';
        ctx.fillRect(0, 0, cw, bandH);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        vCurve.forEach((val, i) => {
          const px = (i / (vCurve.length - 1)) * cw;
          const py = bandH - (val / maxV) * (bandH - 5);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }
      if (hCurve && hCurve.length > 1) {
        const maxH = Math.max(1, ...hCurve);
        const bandW = Math.min(46, cw * 0.055);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.48)';
        ctx.fillRect(0, 0, bandW, ch);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.beginPath();
        hCurve.forEach((val, i) => {
          const py = (i / (hCurve.length - 1)) * ch;
          const px = bandW - (val / maxH) * (bandW - 5);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }
      ctx.restore();
    }

    if (showOverlay && analysis.boxes.length > 0) {
      const strokeW = Math.max(2, Math.round(cw / 650));
      const fontSize = Math.max(12, Math.round(cw / 85));

      analysis.boxes.forEach((box: SplitBox, idx: number) => {
        const isSel = idx === selectedIndex;
        const bx = box.x * scale;
        const by = box.y * scale;
        const bw = box.width * scale;
        const bh = box.height * scale;

        ctx.save();
        ctx.fillStyle = isSel ? 'rgba(37, 99, 235, 0.16)' : 'rgba(37, 99, 235, 0.05)';
        ctx.fillRect(bx, by, bw, bh);

        ctx.strokeStyle = isSel ? '#d97706' : '#2563eb';
        ctx.lineWidth = isSel ? strokeW * 1.6 : strokeW;
        ctx.strokeRect(bx, by, bw, bh);

        const label = `#${String(box.index).padStart(2, '0')} · ${box.width}×${box.height}`;
        ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
        const textW = ctx.measureText(label).width;
        const pillW = textW + 12;
        const pillH = fontSize + 10;
        const lx = Math.min(cw - pillW - 4, Math.max(4, bx + 6));
        const ly = Math.min(ch - pillH - 4, Math.max(4, by + 6));

        ctx.fillStyle = isSel ? '#d97706' : '#0f172a';
        ctx.fillRect(lx, ly, pillW, pillH);
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, lx + 6, ly + pillH / 2);
        ctx.restore();
      });
    }
  }, [loadedSrc, analysis, selectedIndex, showOverlay, showSeamEnergy]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !analysis) return;
    const rect = canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * analysis.imageWidth;
    const py = ((e.clientY - rect.top) / rect.height) * analysis.imageHeight;
    const hitIdx = analysis.boxes.findIndex(
      (b) => px >= b.x && px <= b.x + b.width && py >= b.y && py <= b.y + b.height
    );
    if (hitIdx >= 0) {
      onSelectBox(hitIdx);
    }
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !analysis) return;
    const rect = canvas.getBoundingClientRect();
    const px = Math.max(0, Math.min(analysis.imageWidth, Math.round(((e.clientX - rect.left) / rect.width) * analysis.imageWidth)));
    const py = Math.max(0, Math.min(analysis.imageHeight, Math.round(((e.clientY - rect.top) / rect.height) * analysis.imageHeight)));
    setHoverCoord({ x: px, y: py });
  };

  const selectedBox = selectedIndex >= 0 && analysis ? analysis.boxes[selectedIndex] : null;

  return (
    <section className="space-y-4">
      {/* 核心性能指标条 */}
      <div className="grid grid-cols-2 divide-x divide-slate-200 rounded-lg border border-slate-200 bg-white sm:grid-cols-5">
        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-slate-500">检测模式</div>
          <div className="mt-0.5 font-mono text-sm font-semibold text-slate-900">
            {analysis ? analysis.detectedMode : '--'}
          </div>
        </div>
        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-slate-500">计算引擎</div>
          <div className="mt-0.5 truncate font-mono text-sm font-semibold text-slate-900">
            {analysis ? analysis.backend : '--'}
          </div>
        </div>
        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-slate-500">解析耗时</div>
          <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-slate-900">
            {analysis ? `${analysis.elapsedMs.toFixed(1)} ms` : '--'}
          </div>
        </div>
        <div className="px-4 py-3">
          <div className="text-[11px] font-medium text-slate-500">裁切编码</div>
          <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-slate-900">
            {analysis ? `${analysis.encodeMs.toFixed(1)} ms` : '--'}
          </div>
        </div>
        <div className="col-span-2 px-4 py-3 sm:col-span-1">
          <div className="text-[11px] font-medium text-slate-500">切片数量</div>
          <div className="mt-0.5 font-mono text-sm font-semibold tabular-nums text-slate-900">
            {analysis ? `${analysis.boxes.length} 个切片` : '--'}
          </div>
        </div>
      </div>

      {/* 预览控制栏 */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-base font-semibold text-slate-900">切分边界预览</h1>
          <p className="mt-0.5 font-mono text-xs tabular-nums text-slate-500">
            {analysis
              ? `${analysis.imageWidth}×${analysis.imageHeight} px · 原图 ${analysis.peakBufferMb.toFixed(2)} MB · 点击任意区域查看详情`
              : '正在解析图像版面布局...'}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-700">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => onToggleOverlay(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 accent-blue-600"
            />
            <span>显示切分框</span>
          </label>

          <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-slate-700">
            <input
              type="checkbox"
              checked={showSeamEnergy}
              onChange={(e) => onToggleSeamEnergy(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 accent-blue-600"
            />
            <span>边缘能量曲线</span>
          </label>

          <label className="flex items-center gap-2 text-xs font-medium text-slate-700">
            <span className="sr-only">导出格式</span>
            <select
              aria-label="导出图片格式"
              value={format}
              onChange={(e) => onChangeFormat(e.target.value as ExportFormat)}
              className="rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-800 focus:border-blue-600 focus:outline-none"
            >
              <option value="png">PNG 无损</option>
              <option value="jpg">JPG 质量 95</option>
              <option value="webp">WebP</option>
            </select>
          </label>
        </div>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-medium text-rose-700">
          {error}
        </div>
      )}

      {/* 画布容器 */}
      <div className="relative overflow-hidden rounded-lg border border-slate-200 bg-white p-3">
        {(!analysis || !sourceImageUrl) && (
          <div className="flex h-80 items-center justify-center text-xs font-medium text-slate-500">
            正在调用 Python 3.12 OpenCV + NumPy 执行向量化解析...
          </div>
        )}
        <canvas
          ref={canvasRef}
          data-testid="preview-canvas"
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMove}
          onMouseLeave={() => setHoverCoord(null)}
          className={`mx-auto max-h-[540px] w-auto max-w-full cursor-crosshair rounded border border-slate-100 ${
            !analysis || !sourceImageUrl ? 'hidden' : 'block'
          } ${loading ? 'opacity-65 transition-opacity duration-150' : ''}`}
        />

        {(hoverCoord || selectedBox) && (
          <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2.5 font-mono text-[11px] tabular-nums text-slate-600">
            <span>
              {selectedBox
                ? `已选中 #${String(selectedBox.index).padStart(2, '0')}：${formatBoxMeta(selectedBox)} · 置信度 ${(
                    selectedBox.confidence * 100
                  ).toFixed(0)}%`
                : '点击任意边界框可查看精确坐标与边缘去污染信息'}
            </span>
            <span>{hoverCoord ? `光标坐标: (${hoverCoord.x}, ${hoverCoord.y})` : ''}</span>
          </div>
        )}
      </div>
    </section>
  );
}
