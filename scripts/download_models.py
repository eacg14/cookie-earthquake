#!/usr/bin/env python3
"""Explicit local setup step: downloads the MediaPipe PoseLandmarker .task
model asset into models/. Defaults to the "full" variant, which trades some
speed for accuracy -- appropriate here since this app processes a handful of
stills offline rather than a real-time video stream, and golf swing poses
(side-on, club/arm occlusion) are exactly the hard case where a lighter
model degrades most.

app.py also calls flipbook.model_assets.ensure_model() itself on first use,
so this script isn't strictly required (e.g. on Streamlit Community Cloud,
where there's no shell to run it) -- but running it ahead of time avoids a
first-load delay while developing locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flipbook.model_assets import MODEL_URLS, ensure_model  # noqa: E402


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
        dest = ensure_model(args.variant, force=args.force)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Model ready at {dest} ({dest.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
