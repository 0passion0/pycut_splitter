import type { AnalyzeParams, BenchmarkEntry, SplitBox } from '../types/splitter';

export function parseTargetDimension(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const num = Number.parseInt(trimmed, 10);
  if (!Number.isFinite(num) || num <= 0) return null;
  return num;
}

export function validateUniformTarget(
  uniformSize: boolean,
  targetWidth: string,
  targetHeight: string
): { valid: boolean; targetWidthNum: number | null; targetHeightNum: number | null; error: string | null } {
  if (!uniformSize) {
    return { valid: true, targetWidthNum: null, targetHeightNum: null, error: null };
  }
  const hasW = targetWidth.trim().length > 0;
  const hasH = targetHeight.trim().length > 0;
  if (hasW !== hasH) {
    return {
      valid: false,
      targetWidthNum: null,
      targetHeightNum: null,
      error: '请同时填写目标宽度和目标高度，或全部留空以使用自动最大尺寸。',
    };
  }
  if (!hasW && !hasH) {
    return { valid: true, targetWidthNum: null, targetHeightNum: null, error: null };
  }
  const w = parseTargetDimension(targetWidth);
  const h = parseTargetDimension(targetHeight);
  if (w === null || h === null || w > 16384 || h > 16384) {
    return {
      valid: false,
      targetWidthNum: null,
      targetHeightNum: null,
      error: '目标尺寸必须为 1 到 16384 之间的整数。',
    };
  }
  return { valid: true, targetWidthNum: w, targetHeightNum: h, error: null };
}

export function buildSliceFilename(sourceName: string, index: number, format: string): string {
  const stem = sourceName.replace(/\.[^.]+$/, '') || 'image';
  const ext = format === 'jpeg' ? 'jpg' : format;
  return `${stem}_slice_${String(index).padStart(2, '0')}.${ext}`;
}

export function formatBoxMeta(box: SplitBox): string {
  const [tt, tr, tb, tl] = box.trimmedEdges;
  return `(${box.x}, ${box.y}) · ${box.width}×${box.height} · 修边 ${tt}/${tr}/${tb}/${tl}`;
}

export function generatePythonSnippet(params: AnalyzeParams, sourceName: string): string {
  const lines: string[] = [
    'from pycut_splitter import SmartSplitter',
    '',
    'splitter = SmartSplitter(',
    `    mode="${params.mode}",`,
    `    sensitivity=${params.sensitivity.toFixed(2)},`,
    `    min_region_ratio=${params.minRegionRatio.toFixed(2)},`,
    `    bleed_trim=${params.bleedTrim},`,
    `    contour_gap_px=${params.contourGapPx},`,
  ];

  if (params.mode === 'grid' && params.gridRefine) {
    lines.push('    grid_refine=True,');
  }

  const { valid, targetWidthNum, targetHeightNum } = validateUniformTarget(
    params.uniformSize,
    params.targetWidth,
    params.targetHeight
  );
  if (params.uniformSize) {
    lines.push('    uniform_size=True,');
    lines.push(`    uniform_mode="${params.uniformMode}",`);
    if (valid && targetWidthNum && targetHeightNum) {
      lines.push(`    target_size=(${targetWidthNum}, ${targetHeightNum}),`);
    }
  }
  lines.push(')');
  lines.push('');

  if (params.mode === 'grid') {
    lines.push(
      `result, paths = splitter.split_file("${sourceName}", "./slices", fmt="${params.format}", rows=${params.rows}, cols=${params.cols}, gutter_px=${params.gutterPx})`
    );
  } else {
    lines.push(`result, paths = splitter.split_file("${sourceName}", "./slices", fmt="${params.format}")`);
  }
  lines.push('print(f"检测到 {len(result.boxes)} 个切片 [{result.detected_mode}]，耗时 {result.elapsed_ms:.2f}ms")');
  return lines.join('\n');
}

export function summarizeBenchmark(entries: BenchmarkEntry[]): {
  avgThroughputMp: number;
  largestPeakRatio: number;
  allAccurate: boolean;
} {
  if (!entries.length) {
    return { avgThroughputMp: 0, largestPeakRatio: 0, allAccurate: true };
  }
  const avgThroughputMp =
    entries.reduce((acc, item) => acc + item.throughputMpPerSec, 0) / entries.length;
  const largest = entries.reduce((best, item) =>
    item.megapixels >= best.megapixels ? item : best
  );
  const largestPeakRatio =
    largest.rawImageMb > 0 ? largest.peakTransientMb / largest.rawImageMb : 0;
  const allAccurate = entries.every((item) => item.slices === item.expectedSlices);
  return {
    avgThroughputMp: Number(avgThroughputMp.toFixed(1)),
    largestPeakRatio: Number(largestPeakRatio.toFixed(3)),
    allAccurate,
  };
}
