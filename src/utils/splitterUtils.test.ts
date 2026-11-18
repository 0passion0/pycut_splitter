import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildSliceFilename,
  formatBoxMeta,
  generatePythonSnippet,
  summarizeBenchmark,
  validateUniformTarget,
} from './splitterUtils';
import type { AnalyzeParams, BenchmarkEntry, SplitBox } from '../types/splitter';

test('validateUniformTarget rejects partial target dimensions', () => {
  const partial = validateUniformTarget(true, '512', '');
  assert.equal(partial.valid, false);
  assert.ok(partial.error);

  const ok = validateUniformTarget(true, '512', '256');
  assert.equal(ok.valid, true);
  assert.equal(ok.targetWidthNum, 512);
  assert.equal(ok.targetHeightNum, 256);

  const autoMax = validateUniformTarget(true, '', '');
  assert.equal(autoMax.valid, true);
  assert.equal(autoMax.targetWidthNum, null);
});

test('buildSliceFilename pads index and strips original extension', () => {
  assert.equal(buildSliceFilename('sample_slides.png', 3, 'png'), 'sample_slides_slice_03.png');
  assert.equal(buildSliceFilename('deck.v2.webp', 12, 'jpg'), 'deck.v2_slice_12.jpg');
});

test('generatePythonSnippet reflects uniform sizing and grid params', () => {
  const params: AnalyzeParams = {
    mode: 'grid',
    sensitivity: 0.7,
    minRegionRatio: 0.08,
    bleedTrim: 2,
    contourGapPx: 8,
    rows: 3,
    cols: 3,
    gutterPx: 12,
    uniformSize: true,
    uniformMode: 'resize',
    targetWidth: '400',
    targetHeight: '300',
    format: 'webp',
  };
  const code = generatePythonSnippet(params, 'board.png');
  assert.ok(code.includes('mode="grid"'));
  assert.ok(code.includes('uniform_mode="resize"'));
  assert.ok(code.includes('target_size=(400, 300)'));
  assert.ok(code.includes('rows=3, cols=3, gutter_px=12'));
});

test('formatBoxMeta and summarizeBenchmark compute accurate metrics', () => {
  const box: SplitBox = {
    index: 1,
    row: 0,
    col: 0,
    x: 10,
    y: 20,
    width: 300,
    height: 200,
    confidence: 0.98,
    strategy: 'xycut',
    trimmedEdges: [1, 2, 0, 1],
  };
  assert.equal(formatBoxMeta(box), '(10, 20) · 300×200 · 修边 1/2/0/1');

  const entries: BenchmarkEntry[] = [
    {
      label: '4K',
      width: 3840,
      height: 2160,
      megapixels: 8.29,
      mode: 'xycut',
      slices: 12,
      expectedSlices: 12,
      analyzeMs: 10.0,
      fullPipelineMs: 45.0,
      throughputMpPerSec: 829.0,
      peakTransientMb: 4.5,
      rawImageMb: 23.7,
      encodedBytes: 120000,
    },
  ];
  const summary = summarizeBenchmark(entries);
  assert.equal(summary.avgThroughputMp, 829.0);
  assert.equal(summary.allAccurate, true);
  assert.ok(summary.largestPeakRatio < 0.25);
});
