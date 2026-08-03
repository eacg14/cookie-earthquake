#!/usr/bin/env python3
"""Downloads the MediaPipe PoseLandmarker .task model asset into models/.

The mediapipe pip package does not bundle this file, so it must be fetched
once before the app can run. Defaults to the "full" variant, which trades
some speed for accuracy -- appropriate here since this app processes a
handful of stills offline rather than a real-time video stream, and golf
swing poses (side-on, club/arm occlusion) are exactly the hard case where a
lighter model degrades most.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download(variant: str, force: bool = False) -> Path:
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {sorted(MODEL_URLS)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if dest.exists() and not force:
        print(f"{dest} already exists, skipping download (use --force to re-download).")
        return dest

    url = MODEL_URLS[variant]
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(MODEL_URLS),
        default="full",
        help="Model size/accuracy tradeoff (default: full)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists")
    args = parser.parse_args()

    try:
        download(args.variant, force=args.force)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
