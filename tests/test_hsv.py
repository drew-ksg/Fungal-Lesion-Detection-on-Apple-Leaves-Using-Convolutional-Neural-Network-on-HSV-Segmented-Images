import numpy as np
import pytest

from src.hsv.segment import (
    apply_mask,
    candidate_boxes,
    segment_leaf,
    synthetic_leaf,
    threshold_mask,
    to_hsv,
)
from src.hsv.thresholds import DISEASE_RANGES, HSVRange


def test_threshold_needs_all_three_channels():
    # One pixel: hue 20, saturation 200, value 200 (orange).
    hsv = np.array([[[20, 200, 200]]], np.uint8)
    inside = HSVRange((10, 100, 100), (30, 255, 255))
    low_sat = HSVRange((10, 210, 100), (30, 255, 255))   # saturation out of range
    assert threshold_mask(hsv, [inside])[0, 0] == 255
    assert threshold_mask(hsv, [low_sat])[0, 0] == 0


def test_invalid_range_rejected():
    with pytest.raises(ValueError):
        HSVRange((0, 0, 0), (200, 255, 255))   # hue tops out at 179 in OpenCV


def test_apply_mask_keeps_only_white_pixels():
    img = np.full((4, 4, 3), 100, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 255
    out = apply_mask(img, mask)
    assert out[0, 0].tolist() == [100, 100, 100]
    assert out[1, 1].tolist() == [0, 0, 0]


def test_candidate_boxes_skip_small_regions():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 255    # 100 px region, kept
    mask[40, 40] = 255          # 1 px speck, dropped
    boxes = candidate_boxes(mask, min_area=30, pad=0)
    assert boxes == [(10, 10, 10, 10)]


def test_synthetic_rust_spots_are_found():
    result = segment_leaf(synthetic_leaf())
    assert len(result.boxes) >= 1
    rust = result.per_disease["cedar_apple_rust"]
    assert rust.max() == 255
    # Background (gray) must never be flagged.
    assert result.lesion_mask[0, 0] == 0


def test_to_hsv_rejects_grayscale():
    with pytest.raises(ValueError):
        to_hsv(np.zeros((5, 5), np.uint8))


def test_all_disease_ranges_valid():
    assert set(DISEASE_RANGES) == {"cedar_apple_rust", "apple_scab", "black_rot"}
