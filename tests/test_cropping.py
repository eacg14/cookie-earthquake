import numpy as np
import pytest

from flipbook.cropping import (
    TIGHT_CROP_WARNING_THRESHOLD,
    is_tight_crop,
    out_of_bounds_fraction,
    warp_and_crop,
    warp_frame,
)
from flipbook.presets import PresetSpec

SMALL_PRESET = PresetSpec(
    key="small",
    label="small",
    platform="test",
    aspect_w=1,
    aspect_h=1,
    output_width=500,
    output_height=500,
)


def test_out_of_bounds_fraction_fully_covered():
    # dst = src - 250 => src corners for a 500x500 canvas are [250,750]x[250,750],
    # entirely inside a 1000x1000 source.
    matrix = np.array([[1.0, 0.0, -250.0], [0.0, 1.0, -250.0]])
    fraction = out_of_bounds_fraction(matrix, SMALL_PRESET, (1000, 1000))
    assert fraction == pytest.approx(0.0, abs=1e-9)


def test_out_of_bounds_fraction_reaches_past_edge():
    # dst = src - 900 => bottom-right dst corner (500,500) maps to src (1400,1400),
    # 400px past a 1000x1000 source's edge => 400/1000 = 0.4 overhang.
    matrix = np.array([[1.0, 0.0, -900.0], [0.0, 1.0, -900.0]])
    fraction = out_of_bounds_fraction(matrix, SMALL_PRESET, (1000, 1000))
    assert fraction == pytest.approx(0.4, abs=1e-6)


def test_is_tight_crop_threshold():
    assert not is_tight_crop(0.0)
    assert not is_tight_crop(TIGHT_CROP_WARNING_THRESHOLD - 0.001)
    assert is_tight_crop(TIGHT_CROP_WARNING_THRESHOLD + 0.001)


def test_warp_frame_output_shape():
    source = np.zeros((1000, 1000, 3), dtype=np.uint8)
    source[:] = (10, 20, 30)
    matrix = np.array([[1.0, 0.0, -250.0], [0.0, 1.0, -250.0]])
    output = warp_frame(source, matrix, SMALL_PRESET)
    assert output.shape == (SMALL_PRESET.output_height, SMALL_PRESET.output_width, 3)


def test_warp_and_crop_reports_fraction():
    source = np.zeros((1000, 1000, 3), dtype=np.uint8)
    matrix = np.array([[1.0, 0.0, -900.0], [0.0, 1.0, -900.0]])
    output, fraction = warp_and_crop(source, matrix, SMALL_PRESET)
    assert output.shape == (SMALL_PRESET.output_height, SMALL_PRESET.output_width, 3)
    assert fraction == pytest.approx(0.4, abs=1e-6)
