import pytest

from flipbook.presets import PRESETS, PresetSpec, get_preset, presets_for_platform


def test_all_presets_have_matching_aspect_and_resolution():
    for preset in PRESETS.values():
        assert preset.output_width / preset.output_height == pytest.approx(preset.aspect_ratio)


def test_v1_instagram_presets_present():
    for key in ("ig_feed_square", "ig_feed_portrait", "ig_stories"):
        assert key in PRESETS
        assert PRESETS[key].platform == "instagram"


def test_get_preset_unknown_key_raises():
    with pytest.raises(KeyError):
        get_preset("does_not_exist")


def test_presets_for_platform():
    instagram_presets = presets_for_platform("instagram")
    assert len(instagram_presets) == 3
    assert presets_for_platform("tiktok") == []


def test_preset_rejects_mismatched_aspect_and_resolution():
    with pytest.raises(ValueError):
        PresetSpec(
            key="bad",
            label="bad",
            platform="instagram",
            aspect_w=1,
            aspect_h=1,
            output_width=100,
            output_height=200,
        )
