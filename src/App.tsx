import { useCallback, useEffect, useRef, useState } from 'react';
import JSZip from 'jszip';
import { BenchmarkPanel } from './components/BenchmarkPanel';
import { LightboxModal } from './components/LightboxModal';
import { ParameterSidebar } from './components/ParameterSidebar';
import { PreviewStage } from './components/PreviewStage';
import { SlicesGallery } from './components/SlicesGallery';
import { TopBar, type ActiveView } from './components/TopBar';
import type {
  AnalyzeParams,
  AnalyzeResponse,
  BenchmarkResponse,
  PresetKey,
} from './types/splitter';
import { buildSliceFilename, validateUniformTarget } from './utils/splitterUtils';

const INITIAL_PARAMS: AnalyzeParams = {
  mode: 'auto',
  sensitivity: 0.65,
  minRegionRatio: 0.08,
  bleedTrim: 2,
  contourGapPx: 8,
  rows: 3,
  cols: 3,
  gutterPx: 0,
  uniformSize: false,
  uniformMode: 'pad',
  targetWidth: '',
  targetHeight: '',
  format: 'png',
};

const PRESET_FILENAMES: Record<PresetKey, string> = {
  slides: 'sample_slides.png',
  bento: 'bento_preset.png',
  gutters: 'gutters_preset.png',
  contours: 'contours_preset.png',
  large_4k: '4k_stress_collage.png',
};

export default function App() {
  const [activeView, setActiveView] = useState<ActiveView>('studio');
  const [activePreset, setActivePreset] = useState<PresetKey | null>('slides');
  const [uploadedBase64, setUploadedBase64] = useState<string | null>(null);
  const [sourceName, setSourceName] = useState<string>('sample_slides.png');
  const [sourcePreviewUrl, setSourcePreviewUrl] = useState<string | null>('/examples/sample_slides.png');

  const [params, setParams] = useState<AnalyzeParams>(INITIAL_PARAMS);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedIndex, setSelectedIndex] = useState<number>(-1);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [showSeamEnergy, setShowSeamEnergy] = useState<boolean>(false);
  const [lightboxIndex, setLightboxIndex] = useState<number>(-1);
  const [exportingZip, setExportingZip] = useState<boolean>(false);

  const [benchmarkData, setBenchmarkData] = useState<BenchmarkResponse | null>(null);
  const [benchmarkRunning, setBenchmarkRunning] = useState<boolean>(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const debounceRef = useRef<number | null>(null);

  const runAnalysis = useCallback(
    async (
      nextParams: AnalyzeParams,
      preset: PresetKey | null,
      imageB64: string | null,
      needsPreview: boolean
    ) => {
      const targetCheck = validateUniformTarget(
        nextParams.uniformSize,
        nextParams.targetWidth,
        nextParams.targetHeight
      );
      if (!targetCheck.valid) {
        return;
      }

      if (abortRef.current) {
        abortRef.current.abort();
      }
      const controller = new AbortController();
      abortRef.current = controller;

      setLoading(true);
      setError(null);

      try {
        const resp = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            preset: imageB64 ? undefined : preset ?? 'slides',
            imageBase64: imageB64 ?? undefined,
            mode: nextParams.mode,
            sensitivity: nextParams.sensitivity,
            minRegionRatio: nextParams.minRegionRatio,
            bleedTrim: nextParams.bleedTrim,
            contourGapPx: nextParams.contourGapPx,
            rows: nextParams.rows,
            cols: nextParams.cols,
            gutterPx: nextParams.gutterPx,
            uniformSize: nextParams.uniformSize,
            uniformMode: nextParams.uniformMode,
            targetWidth: targetCheck.targetWidthNum,
            targetHeight: targetCheck.targetHeightNum,
            format: nextParams.format,
            includeTiles: true,
            includeSourcePreview: needsPreview,
          }),
        });

        if (!resp.ok) {
          const errBody = await resp.json().catch(() => ({}));
          throw new Error(errBody.error || `服务端错误 (${resp.status})`);
        }

        const data = (await resp.json()) as AnalyzeResponse;
        setAnalysis(data);
        if (data.sourcePreviewUrl) {
          setSourcePreviewUrl(data.sourcePreviewUrl);
        }
        setSelectedIndex((prev) => (prev >= 0 && prev < data.boxes.length ? prev : -1));
      } catch (err) {
        if ((err as Error).name === 'AbortError') return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (abortRef.current === controller) {
          setLoading(false);
        }
      }
    },
    []
  );

  // Initial load of sample_slides.png
  useEffect(() => {
    runAnalysis(INITIAL_PARAMS, 'slides', null, false);
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [runAnalysis]);

  const scheduleAnalysis = useCallback(
    (nextParams: AnalyzeParams, preset: PresetKey | null, imageB64: string | null, needsPreview: boolean) => {
      if (debounceRef.current) {
        window.clearTimeout(debounceRef.current);
      }
      debounceRef.current = window.setTimeout(() => {
        runAnalysis(nextParams, preset, imageB64, needsPreview);
      }, 90);
    },
    [runAnalysis]
  );

  const handleUpdateParams = (patch: Partial<AnalyzeParams>) => {
    const nextParams = { ...params, ...patch };
    setParams(nextParams);
    scheduleAnalysis(nextParams, activePreset, uploadedBase64, false);
  };

  const handleSelectPreset = (preset: PresetKey) => {
    setActiveView('studio');
    setActivePreset(preset);
    setUploadedBase64(null);
    setSelectedIndex(-1);
    setLightboxIndex(-1);
    setSourceName(PRESET_FILENAMES[preset]);
    if (preset === 'slides') {
      setSourcePreviewUrl('/examples/sample_slides.png');
      runAnalysis(params, preset, null, false);
    } else {
      setSourcePreviewUrl(null);
      setAnalysis(null);
      runAnalysis(params, preset, null, true);
    }
  };

  const handleUploadFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('请选择有效的 PNG、JPG 或 WebP 图片文件。');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || '');
      setActiveView('studio');
      setActivePreset(null);
      setUploadedBase64(dataUrl);
      setSourceName(file.name);
      setSourcePreviewUrl(dataUrl);
      setSelectedIndex(-1);
      setLightboxIndex(-1);
      runAnalysis(params, null, dataUrl, false);
    };
    reader.readAsDataURL(file);
  };

  const handleDownloadTile = (idx: number) => {
    if (!analysis) return;
    const box = analysis.boxes[idx];
    const tile = analysis.tiles.find((t) => t.index === box?.index);
    if (!box || !tile) return;

    const a = document.createElement('a');
    a.href = tile.dataUrl;
    a.download = buildSliceFilename(sourceName, box.index, params.format);
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const handleExportZip = async () => {
    if (!analysis || !analysis.tiles.length) return;
    setExportingZip(true);
    try {
      const zip = new JSZip();
      for (const tile of analysis.tiles) {
        const b64 = tile.dataUrl.split(',')[1] || '';
        const filename = buildSliceFilename(sourceName, tile.index, params.format);
        zip.file(filename, b64, { base64: true });
      }
      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const stem = sourceName.replace(/\.[^.]+$/, '') || 'pycut';
      const a = document.createElement('a');
      a.href = url;
      a.download = `${stem}_slices.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } finally {
      setExportingZip(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkRunning(true);
    setBenchmarkError(null);
    try {
      const resp = await fetch('/api/benchmark', { method: 'POST' });
      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}));
        throw new Error(errBody.error || `基准测试运行失败 (${resp.status})`);
      }
      const data = (await resp.json()) as BenchmarkResponse;
      setBenchmarkData(data);
    } catch (err) {
      setBenchmarkError(err instanceof Error ? err.message : String(err));
    } finally {
      setBenchmarkRunning(false);
    }
  };

  const handleSelectBenchmarks = () => {
    setActiveView('benchmarks');
    if (!benchmarkData && !benchmarkRunning) {
      handleRunBenchmark();
    }
  };

  const lightboxBox =
    lightboxIndex >= 0 && analysis ? analysis.boxes[lightboxIndex] ?? null : null;
  const lightboxTile =
    lightboxBox && analysis
      ? analysis.tiles.find((t) => t.index === lightboxBox.index) ?? null
      : null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <TopBar
        activeView={activeView}
        activePreset={activePreset}
        sliceCount={analysis?.boxes.length ?? 0}
        exportingZip={exportingZip}
        onSelectPreset={handleSelectPreset}
        onSelectBenchmarks={handleSelectBenchmarks}
        onExportZip={handleExportZip}
      />

      {activeView === 'benchmarks' ? (
        <main className="mx-auto max-w-6xl px-6 py-8">
          <BenchmarkPanel
            data={benchmarkData}
            running={benchmarkRunning}
            error={benchmarkError}
            onRunBenchmark={handleRunBenchmark}
          />
        </main>
      ) : (
        <div className="flex min-h-[calc(100vh-56px)] flex-col lg:flex-row">
          <ParameterSidebar
            params={params}
            sourceName={sourceName}
            sourceWidth={analysis?.imageWidth ?? 0}
            sourceHeight={analysis?.imageHeight ?? 0}
            onUpdateParams={handleUpdateParams}
            onUploadFile={handleUploadFile}
            onLoad4KStress={() => handleSelectPreset('large_4k')}
          />

          <main className="min-w-0 flex-1 space-y-8 p-6 lg:p-8">
            <PreviewStage
              sourceImageUrl={sourcePreviewUrl}
              analysis={analysis}
              loading={loading}
              error={error}
              selectedIndex={selectedIndex}
              showOverlay={showOverlay}
              showSeamEnergy={showSeamEnergy}
              format={params.format}
              onToggleOverlay={setShowOverlay}
              onToggleSeamEnergy={setShowSeamEnergy}
              onChangeFormat={(fmt) => handleUpdateParams({ format: fmt })}
              onSelectBox={(idx) => setSelectedIndex(idx)}
            />

            <SlicesGallery
              boxes={analysis?.boxes ?? []}
              tiles={analysis?.tiles ?? []}
              selectedIndex={selectedIndex}
              format={params.format}
              onSelectBox={(idx) => setSelectedIndex(idx)}
              onInspectTile={(idx) => {
                setSelectedIndex(idx);
                setLightboxIndex(idx);
              }}
              onDownloadTile={handleDownloadTile}
            />
          </main>
        </div>
      )}

      <LightboxModal
        open={lightboxIndex >= 0}
        box={lightboxBox}
        tile={lightboxTile}
        onClose={() => setLightboxIndex(-1)}
        onDownload={() => {
          if (lightboxIndex >= 0) handleDownloadTile(lightboxIndex);
        }}
      />
    </div>
  );
}
