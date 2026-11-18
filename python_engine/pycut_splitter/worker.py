"""Persistent JSON-line IPC worker for PyCut Studio backend."""

from __future__ import annotations

import base64
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure python_engine root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycut_splitter import SmartSplitter, batch_crop_to_memory_py
from pycut_splitter.memory_safe import decode_image_safe


def _decode_b64_image(b64_data: str) -> np.ndarray:
    if "," in b64_data and b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]
    raw = np.frombuffer(base64.b64decode(b64_data), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Unable to decode uploaded image bytes.")
    if decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    if decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _generate_synthetic_preset(name: str) -> np.ndarray:
    if name == "bento":
        img = np.full((800, 1200, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (594, 800), (24, 52, 110), -1)
        cv2.rectangle(img, (597, 0), (1200, 396), (215, 72, 65), -1)
        cv2.rectangle(img, (597, 399), (1200, 800), (32, 145, 112), -1)
        cv2.putText(img, "Panel A (Tall)", (44, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.putText(img, "Panel B (Top Right)", (636, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(img, "Panel C (Bottom Right)", (636, 476), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return img

    if name == "gutters":
        img = np.full((760, 1000, 3), 255, dtype=np.uint8)
        cards = [
            (36, 36, 480, 360, (30, 64, 175), "Card 01"),
            (520, 36, 964, 360, (5, 150, 105), "Card 02"),
            (36, 400, 480, 724, (190, 18, 60), "Card 03"),
            (520, 400, 964, 724, (217, 119, 6), "Card 04"),
        ]
        for x0, y0, x1, y1, col, lbl in cards:
            cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
            cv2.putText(img, lbl, (x0 + 28, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        return img

    if name == "large_4k":
        # 3840x2160 4K 12-card stress-test collage with subtle 1px hairlines
        img = np.full((2160, 3840, 3), 248, dtype=np.uint8)
        palette = [
            (24, 52, 110), (15, 118, 110), (185, 28, 28), (30, 64, 175),
            (6, 95, 70), (180, 83, 9), (67, 56, 202), (190, 24, 93),
            (3, 105, 161), (21, 128, 61), (126, 34, 206), (51, 65, 85),
        ]
        cols, rows = 4, 3
        cell_w = 3840 // cols
        cell_h = 2160 // rows
        idx = 0
        for r in range(rows):
            for c in range(cols):
                x0 = c * cell_w + 18
                y0 = r * cell_h + 18
                x1 = (c + 1) * cell_w - 18
                y1 = (r + 1) * cell_h - 18
                cv2.rectangle(img, (x0, y0), (x1, y1), palette[idx % len(palette)], -1)
                cv2.putText(
                    img,
                    f"4K Tile #{idx + 1:02d}",
                    (x0 + 48, y0 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (255, 255, 255),
                    4,
                )
                idx += 1
        return img

    # Default: contours
    img = np.full((950, 1500, 3), 248, dtype=np.uint8)
    blobs = [
        (80, 70, 400, 350, (29, 78, 216), "Object #1"),
        (540, 90, 920, 330, (4, 120, 87), "Object #2"),
        (1060, 65, 1400, 375, (190, 18, 60), "Object #3"),
        (170, 490, 630, 830, (15, 23, 42), "Object #4"),
        (810, 470, 1320, 840, (217, 119, 6), "Object #5"),
    ]
    for x0, y0, x1, y1, col, lbl in blobs:
        cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
        cv2.putText(img, lbl, (x0 + 24, y0 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def _handle_analyze(req: Dict[str, Any]) -> Dict[str, Any]:
    preset = req.get("preset")
    image_b64 = req.get("imageBase64")
    image_path = req.get("imagePath")

    if image_b64:
        img = _decode_b64_image(image_b64)
    elif image_path:
        img = decode_image_safe(image_path)
    elif preset:
        img = _generate_synthetic_preset(str(preset))
    else:
        raise ValueError("Missing imageBase64, imagePath, or preset")

    target_w = req.get("targetWidth")
    target_h = req.get("targetHeight")
    target_size: Optional[Tuple[int, int]] = None
    if target_w and target_h:
        target_size = (int(target_w), int(target_h))

    splitter = SmartSplitter(
        mode=str(req.get("mode", "auto")),
        sensitivity=float(req.get("sensitivity", 0.65)),
        min_region_ratio=float(req.get("minRegionRatio", 0.08)),
        bleed_trim=int(req.get("bleedTrim", 2)),
        contour_gap_px=int(req.get("contourGapPx", 8)),
        uniform_size=bool(req.get("uniformSize", False)),
        target_size=target_size,
        uniform_mode=str(req.get("uniformMode", "pad")),
    )

    rows = int(req.get("rows", 3))
    cols = int(req.get("cols", 3))
    gutter_px = int(req.get("gutterPx", 0))
    fmt = str(req.get("format", "png"))

    t0 = time.perf_counter()
    res = splitter.analyze(img, rows=rows, cols=cols, gutter_px=gutter_px, collect_diagnostics=True)
    analyze_ms = (time.perf_counter() - t0) * 1000.0
    raw_mb = img.nbytes / (1024.0 * 1024.0)

    tiles_b64: List[Dict[str, Any]] = []
    encode_ms = 0.0
    if res.boxes and req.get("includeTiles", True):
        t1 = time.perf_counter()
        coords = [(b.x, b.y, b.width, b.height) for b in res.boxes]
        resolved_target = splitter._resolve_target_size(res.boxes)
        blobs = batch_crop_to_memory_py(
            img,
            coords,
            fmt=fmt,
            target_size=resolved_target,
            uniform_mode=splitter.uniform_mode,
        )
        encode_ms = (time.perf_counter() - t1) * 1000.0
        mime = "image/jpeg" if fmt in ("jpg", "jpeg") else f"image/{fmt}"
        for b, blob in zip(res.boxes, blobs):
            out_w = resolved_target[0] if resolved_target else b.width
            out_h = resolved_target[1] if resolved_target else b.height
            tiles_b64.append(
                {
                    "index": b.index,
                    "width": out_w,
                    "height": out_h,
                    "byteSize": len(blob),
                    "dataUrl": f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}",
                }
            )

    source_preview_url: Optional[str] = None
    if req.get("includeSourcePreview", False):
        # Downscale preview only if >1920px wide for fast UI transport; analysis ran at 100% 1:1 resolution
        h, w = img.shape[:2]
        if w > 1920:
            scale = 1920.0 / float(w)
            prev = cv2.resize(img, (1920, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            prev = img
        ch = prev.shape[2]
        bgr = cv2.cvtColor(prev, cv2.COLOR_RGBA2BGR if ch == 4 else cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            source_preview_url = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"

    return {
        "imageWidth": res.image_width,
        "imageHeight": res.image_height,
        "detectedMode": res.detected_mode,
        "backend": res.backend,
        "elapsedMs": round(analyze_ms, 2),
        "encodeMs": round(encode_ms, 2),
        "peakBufferMb": round(raw_mb, 2),
        "diagnostics": res.diagnostics,
        "sourcePreviewUrl": source_preview_url,
        "boxes": [
            {
                "index": b.index,
                "row": b.row,
                "col": b.col,
                "x": b.x,
                "y": b.y,
                "width": b.width,
                "height": b.height,
                "confidence": round(b.confidence, 4),
                "strategy": b.strategy,
                "trimmedEdges": list(b.trimmed_edges),
            }
            for b in res.boxes
        ],
        "tiles": tiles_b64,
    }


def _handle_benchmark() -> Dict[str, Any]:
    cases = [
        ("1080p Bento 拼图 (1920x1080)", 1920, 1080, "xycut", 3, 2),
        ("2.8K 幻灯片集 (2848x1600)", 2848, 1600, "xycut", 3, 3),
        ("4K 超清网格 (3840x2160)", 3840, 2160, "auto", 4, 3),
        ("4K 轮廓散件 (3840x2160)", 3840, 2160, "contour", 3, 3),
        ("8K 极限压测 (7680x4320)", 7680, 4320, "xycut", 4, 4),
    ]
    results: List[Dict[str, Any]] = []
    splitter_xy = SmartSplitter(mode="xycut", sensitivity=0.65, bleed_trim=2)
    splitter_auto = SmartSplitter(mode="auto", sensitivity=0.65, bleed_trim=2)
    splitter_ct = SmartSplitter(mode="contour", sensitivity=0.70, contour_gap_px=8)

    # Warm up OpenCV SIMD / NumPy kernel caches once before timing
    _warmup_img = np.full((64, 64, 3), 240, dtype=np.uint8)
    _warmup_img[10:54, 10:54] = (40, 80, 160)
    splitter_xy.analyze(_warmup_img)
    splitter_ct.analyze(_warmup_img)
    del _warmup_img

    for label, w, h, mode, cols, rows in cases:
        img = np.full((h, w, 3), 250, dtype=np.uint8)
        cw = w // cols
        ch = h // rows
        for r in range(rows):
            for c in range(cols):
                x0 = c * cw + 12
                y0 = r * ch + 12
                x1 = (c + 1) * cw - 12
                y1 = (r + 1) * ch - 12
                color = (30 + (r * 45) % 180, 70 + (c * 55) % 160, 150)
                img[y0:y1, x0:x1] = color

        sp = splitter_ct if mode == "contour" else (splitter_auto if mode == "auto" else splitter_xy)

        # Warmup once on 1st case or measure pure SIMD latency without tracemalloc overhead
        t0 = time.perf_counter()
        res = sp.analyze(img)
        analyze_ms = (time.perf_counter() - t0) * 1000.0

        tracemalloc.start()
        sp.analyze(img)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t1 = time.perf_counter()
        _, blobs = sp.split_to_bytes(img, fmt="png")
        total_ms = (time.perf_counter() - t1) * 1000.0

        megapixels = (w * h) / 1_000_000.0
        results.append(
            {
                "label": label,
                "width": w,
                "height": h,
                "megapixels": round(megapixels, 2),
                "mode": res.detected_mode,
                "slices": len(res.boxes),
                "expectedSlices": rows * cols,
                "analyzeMs": round(analyze_ms, 2),
                "fullPipelineMs": round(total_ms, 2),
                "throughputMpPerSec": round(megapixels / max(0.0005, analyze_ms / 1000.0), 1),
                "peakTransientMb": round(peak_bytes / (1024.0 * 1024.0), 2),
                "rawImageMb": round((w * h * 3) / (1024.0 * 1024.0), 2),
                "encodedBytes": sum(len(b) for b in blobs),
            }
        )
        del img

    return {
        "cv2Version": cv2.__version__,
        "numpyVersion": np.__version__,
        "results": results,
    }


def main() -> None:
    # Signal readiness to parent Node process
    sys.stdout.write(json.dumps({"ready": True, "cv2": cv2.__version__, "numpy": np.__version__}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            action = payload.get("action")
            if action == "analyze":
                data = _handle_analyze(payload)
            elif action == "benchmark":
                data = _handle_benchmark()
            elif action == "ping":
                data = {"pong": True, "cv2": cv2.__version__, "numpy": np.__version__}
            else:
                raise ValueError(f"Unknown action: {action}")
            resp = {"id": req_id, "ok": True, "data": data}
        except Exception as exc:
            resp = {"id": req_id, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
