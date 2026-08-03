"""Platform/format output presets.

Each preset defines only the output aspect ratio/resolution for a social
platform format. There's no positioning field: the reference frame is used
exactly as shot (a plain center-crop to the target aspect ratio, no
repositioning), and every other frame is aligned to match it via background
feature matching (features.py) -- not by tracking or recentering the
subject. Add new platforms/formats by appending rows to PRESETS -- no other
module needs to change.

Crop size is not a preset field either: pipeline.py computes it
automatically per-series as the largest size that keeps every aligned
frame's real pixel data (no border replication).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresetSpec:
    key: str
    label: str
    platform: str
    aspect_w: int
    aspect_h: int
    output_width: int
    output_height: int

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError(f"{self.key}: output dimensions must be positive")
        if abs(self.output_width / self.output_height - self.aspect_w / self.aspect_h) > 1e-6:
            raise ValueError(f"{self.key}: output resolution does not match aspect ratio")

    @property
    def aspect_ratio(self) -> float:
        return self.aspect_w / self.aspect_h


PRESETS: dict[str, PresetSpec] = {
    p.key: p
    for p in [
        PresetSpec(
            key="ig_feed_square",
            label="Instagram Feed — Square (1:1)",
            platform="instagram",
            aspect_w=1,
            aspect_h=1,
            output_width=1080,
            output_height=1080,
        ),
        PresetSpec(
            key="ig_feed_portrait",
            label="Instagram Feed — Portrait (4:5)",
            platform="instagram",
            aspect_w=4,
            aspect_h=5,
            output_width=1080,
            output_height=1350,
        ),
        PresetSpec(
            key="ig_stories",
            label="Instagram Stories/Reels (9:16)",
            platform="instagram",
            aspect_w=9,
            aspect_h=16,
            output_width=1080,
            output_height=1920,
        ),
    ]
}


def get_preset(key: str) -> PresetSpec:
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown preset key: {key!r}") from exc


def presets_for_platform(platform: str) -> list[PresetSpec]:
    return [p for p in PRESETS.values() if p.platform == platform]
