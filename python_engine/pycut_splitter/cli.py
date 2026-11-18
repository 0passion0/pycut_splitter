"""Command-line interface for PyCut Image Splitter (`pycut-split`)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from .engine import HAS_OPENCV_CORE, SmartSplitter


def _parse_target_size(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    if spec is None:
        return None
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid --target-size '{spec}', expected WIDTHxHEIGHT such as 512x512")
    w = int(parts[0].strip())
    h = int(parts[1].strip())
    if w <= 0 or h <= 0:
        raise ValueError(f"--target-size dimensions must be positive, got {w}x{h}")
    return (w, h)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pycut-split",
        description="Memory-safe, vectorized OpenCV + NumPy composite image splitter.",
    )
    parser.add_argument("input", type=Path, help="Input image path or directory of images")
    parser.add_argument("-o", "--output", type=Path, default=Path("./output_slices"), help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["auto", "xycut", "contour", "grid"],
        default="auto",
        help="Layout detection mode (default: auto)",
    )
    parser.add_argument("--sensitivity", type=float, default=0.65, help="Detection sensitivity (0.1 - 1.0)")
    parser.add_argument("--min-region-ratio", type=float, default=0.08, help="Min region ratio (0.03 - 0.35)")
    parser.add_argument("--bleed-trim", type=int, default=2, help="Bleed decontamination pixels")
    parser.add_argument("--contour-gap", type=int, default=8, help="Contour closing gap in px")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=3, help="Grid columns")
    parser.add_argument("--gutter-px", type=int, default=0, help="Grid gutter pixels")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "webp"], help="Output image format")
    parser.add_argument("--uniform-size", action="store_true", help="Normalize all slices to uniform dimensions")
    parser.add_argument("--target-size", default=None, help="Explicit uniform size WIDTHxHEIGHT, e.g. 512x512")
    parser.add_argument(
        "--uniform-mode",
        default="pad",
        choices=["pad", "resize", "crop"],
        help="Uniform sizing strategy: pad, resize, or crop",
    )

    args = parser.parse_args()
    parsed_target_size = _parse_target_size(args.target_size)

    splitter = SmartSplitter(
        mode=args.mode,
        sensitivity=args.sensitivity,
        min_region_ratio=args.min_region_ratio,
        bleed_trim=args.bleed_trim,
        contour_gap_px=args.contour_gap,
        uniform_size=args.uniform_size,
        target_size=parsed_target_size,
        uniform_mode=args.uniform_mode,
    )

    inputs = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        if args.input.is_dir()
        else [args.input]
    )

    print(f"PyCut Splitter v1.0.0 | OpenCV+NumPy SIMD Core: {'ACTIVE' if HAS_OPENCV_CORE else 'INACTIVE'}")
    for img_path in inputs:
        sub_out = args.output / img_path.stem if args.input.is_dir() else args.output
        result, saved = splitter.split_file(
            img_path,
            sub_out,
            prefix=img_path.stem,
            fmt=args.format,
            rows=args.rows,
            cols=args.cols,
            gutter_px=args.gutter_px,
        )
        print(
            f"  [OK] {img_path.name} ({result.image_width}x{result.image_height}) -> "
            f"{len(saved)} slices [{result.detected_mode} / {result.backend}] in {result.elapsed_ms:.1f}ms"
        )


if __name__ == "__main__":
    main()
