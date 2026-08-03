"""Fetches the MediaPipe PoseLandmarker `.task` model asset.

The mediapipe pip package doesn't bundle this file. `ensure_model` is used
both by scripts/download_models.py (an explicit one-time local setup step)
and by app.py itself, so a freshly deployed instance -- e.g. on Streamlit
Community Cloud, where there's no shell access to pre-run a script -- fetches
it automatically on first use instead of requiring a manual step.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

DEFAULT_VARIANT = "full"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(variant: str = DEFAULT_VARIANT, models_dir: Path | None = None) -> Path:
    models_dir = models_dir or MODELS_DIR
    return models_dir / f"pose_landmarker_{variant}.task"


def ensure_model(
    variant: str = DEFAULT_VARIANT,
    models_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Return the path to the model file, downloading it first if missing."""
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {sorted(MODEL_URLS)}")

    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = model_path(variant, models_dir)
    if dest.exists() and not force:
        return dest

    urllib.request.urlretrieve(MODEL_URLS[variant], dest)
    return dest
