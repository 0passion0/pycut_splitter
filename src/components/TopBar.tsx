import type { PresetKey } from '../types/splitter';

export type ActiveView = 'studio' | 'benchmarks';

interface TopBarProps {
  activeView: ActiveView;
  activePreset: PresetKey | null;
  sliceCount: number;
  exportingZip: boolean;
  onSelectPreset: (preset: PresetKey) => void;
  onSelectBenchmarks: () => void;
  onExportZip: () => void;
}

const PRESET_NAV: { key: PresetKey; label: string }[] = [
  { key: 'slides', label: '3×3 幻灯片' },
  { key: 'bento', label: 'Bento 拼图' },
  { key: 'gutters', label: '留白拼贴' },
  { key: 'contours', label: '轮廓散件' },
];

export function TopBar({
  activeView,
  activePreset,
  sliceCount,
  exportingZip,
  onSelectPreset,
  onSelectBenchmarks,
  onExportZip,
}: TopBarProps) {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-8 border-b border-slate-200 bg-white px-6">
      <a
        href="/"
        onClick={(e) => {
          e.preventDefault();
          onSelectPreset('slides');
        }}
        className="shrink-0 whitespace-nowrap text-base font-bold tracking-tight text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
      >
        PyCut Studio
      </a>

      <nav className="flex items-center gap-6 overflow-x-auto" aria-label="工作区预设与性能基准">
        {PRESET_NAV.map((item) => {
          const isActive = activeView === 'studio' && activePreset === item.key;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onSelectPreset(item.key)}
              className={`shrink-0 whitespace-nowrap border-b-2 py-1.5 text-sm font-medium transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 ${
                isActive
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-slate-600 hover:text-slate-900'
              }`}
            >
              {item.label}
            </button>
          );
        })}
        <button
          type="button"
          onClick={onSelectBenchmarks}
          className={`shrink-0 whitespace-nowrap border-b-2 py-1.5 text-sm font-medium transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 ${
            activeView === 'benchmarks'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-600 hover:text-slate-900'
          }`}
        >
          性能基准
        </button>
      </nav>

      <div className="flex shrink-0 items-center">
        <button
          type="button"
          onClick={onExportZip}
          disabled={sliceCount === 0 || exportingZip}
          className="shrink-0 whitespace-nowrap rounded-md bg-blue-600 px-4 py-2 text-xs font-semibold text-white transition-colors duration-150 hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-45 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
        >
          {exportingZip ? '正在打包 ZIP...' : '导出 ZIP'}
        </button>
      </div>
    </header>
  );
}
