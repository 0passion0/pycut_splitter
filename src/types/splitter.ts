export type SplitMode = 'auto' | 'xycut' | 'contour' | 'grid';
export type UniformMode = 'pad' | 'resize' | 'crop';
export type ExportFormat = 'png' | 'jpg' | 'webp';
export type PresetKey = 'slides' | 'bento' | 'gutters' | 'contours' | 'large_4k';

export interface SplitBox {
  index: number;
  row: number;
  col: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  strategy: string;
  trimmedEdges: [number, number, number, number];
}

export interface EncodedTile {
  index: number;
  width: number;
  height: number;
  byteSize: number;
  dataUrl: string;
}

export interface SeamDiagnostics {
  bg_rgb: [number, number, number];
  is_transparent: boolean;
  bg_uniformity: number;
  seam_curves: {
    h?: number[];
    v?: number[];
  };
}

export interface AnalyzeResponse {
  imageWidth: number;
  imageHeight: number;
  detectedMode: string;
  backend: string;
  elapsedMs: number;
  encodeMs: number;
  peakBufferMb: number;
  diagnostics: SeamDiagnostics | null;
  sourcePreviewUrl: string | null;
  boxes: SplitBox[];
  tiles: EncodedTile[];
}

export interface AnalyzeParams {
  mode: SplitMode;
  sensitivity: number;
  minRegionRatio: number;
  bleedTrim: number;
  contourGapPx: number;
  rows: number;
  cols: number;
  gutterPx: number;
  uniformSize: boolean;
  uniformMode: UniformMode;
  targetWidth: string;
  targetHeight: string;
  format: ExportFormat;
}

export interface BenchmarkEntry {
  label: string;
  width: number;
  height: number;
  megapixels: number;
  mode: string;
  slices: number;
  expectedSlices: number;
  analyzeMs: number;
  fullPipelineMs: number;
  throughputMpPerSec: number;
  peakTransientMb: number;
  rawImageMb: number;
  encodedBytes: number;
}

export interface BenchmarkResponse {
  cv2Version: string;
  numpyVersion: string;
  results: BenchmarkEntry[];
}
