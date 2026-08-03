"""Final encoding of aligned frames: ordered filenames and a downloadable zip."""

from __future__ import annotations

import io
import zipfile

import numpy as np
from PIL import Image

from .models import FramePlan
from .presets import PresetSpec

JPEG_QUALITY = 95


def frame_filename(frame_index: int, total_frames: int, preset: PresetSpec) -> str:
    width = max(2, len(str(total_frames)))
    return f"{frame_index + 1:0{width}d}_{preset.key}.jpg"


def encode_jpeg(image_rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def build_zip(plans: list[FramePlan], preset: PresetSpec, total_frames: int) -> bytes:
    """Zip the rendered output_image of each plan, skipping any without one
    (i.e. still blocked on manual correction). Use pipeline.exportable_plans
    first if you want export to fail loudly on blocked frames instead.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for plan in plans:
            if plan.output_image is None:
                continue
            name = frame_filename(plan.frame_index, total_frames, preset)
            zf.writestr(name, encode_jpeg(plan.output_image))
    return buffer.getvalue()
