"""Platform/format output presets.

Each preset defines the output aspect ratio/resolution for a social platform
format, plus where the alignment anchor (hip midpoint) and scale reference
(torso length) should land within that canvas. Add new platforms/formats by
appending rows to PRESETS -- no other module needs to change.
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
    # Where the anchor (hip midpoint) should land, as a fraction of
    # (output_width, output_height). E.g. (0.5, 0.62) centers horizontally
    # and places the hips a bit below vertical center.
    target_anchor_frac: tuple[float, float]
    # Torso length (shoulder-mid to hip-mid) as a fraction of output_height.
    target_scale_frac: float

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError(f"{self.key}: output dimensions must be positive")
        if abs(self.output_width / self.output_height - self.aspect_w / self.aspect_h) > 1e-6:
            raise ValueError(f"{self.key}: output resolution does not match aspect ratio")
        if not (0.0 < self.target_anchor_frac[0] < 1.0 and 0.0 < self.target_anchor_frac[1] < 1.0):
            raise ValueError(f"{self.key}: target_anchor_frac must be within (0, 1)")
        if not (0.0 < self.target_scale_frac < 1.0):
            raise ValueError(f"{self.key}: target_scale_frac must be within (0, 1)")

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
            target_anchor_frac=(0.5, 0.62),
            target_scale_frac=0.32,
        ),
        PresetSpec(
            key="ig_feed_portrait",
            label="Instagram Feed — Portrait (4:5)",
            platform="instagram",
            aspect_w=4,
            aspect_h=5,
            output_width=1080,
            output_height=1350,
            target_anchor_frac=(0.5, 0.6),
            target_scale_frac=0.28,
        ),
        PresetSpec(
            key="ig_stories",
            label="Instagram Stories/Reels (9:16)",
            platform="instagram",
            aspect_w=9,
            aspect_h=16,
            output_width=1080,
            output_height=1920,
            target_anchor_frac=(0.5, 0.55),
            target_scale_frac=0.22,
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
