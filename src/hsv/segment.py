"""Convert a leaf picture into HSV masks and a masked color image.
made this so that the functions in this file can be imported without opening a window or
depending on a specific path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .thresholds import DISEASE_RANGES, LEAF_RANGES, HSVRange


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image into HSV."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a color image with 3 channels")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def threshold_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    """Create a mask where pixels fall inside at least one HSV range."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in ranges:
        range_mask = np.asarray(
            cv2.inRange(
                hsv,
                np.array(hsv_range.lower, dtype=np.uint8),
                np.array(hsv_range.upper, dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        mask = np.bitwise_or(mask, range_mask)
    return mask


def clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    """Remove small specks and fill small holes in a binary mask."""
    if open_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def leaf_region(hsv: np.ndarray, grow: int = 3) -> np.ndarray:
    """Create a mask for the largest green leaf region in the image."""
    leaf = threshold_mask(hsv, LEAF_RANGES)
    leaf = clean_mask(leaf, open_size=5, close_size=15)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(leaf)
    if count <= 1:
        return leaf

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = np.where(labels == largest, 255, 0).astype(np.uint8)

    # fill holes so lesions inside the leaf are not removed from the leaf mask.
    contours, _ = cv2.findContours(
        leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(leaf)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    if grow > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)
        )
        filled = cv2.dilate(filled, kernel)
    return filled


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep the original pixels where the mask is white."""
    return cv2.bitwise_and(bgr, bgr, mask=mask)

@dataclass
class SegmentationResult:
    """Stores the results produced by the full HSV process."""

    hsv: np.ndarray
    leaf_mask: np.ndarray
    lesion_mask: np.ndarray
    masked: np.ndarray
    per_disease: dict[str, np.ndarray] = field(default_factory=dict)


def segment_leaf(
    bgr: np.ndarray,
    disease_ranges: dict[str, list[HSVRange]] | None = None,
) -> SegmentationResult:
    """Combines HSV ranges and return a masked BGR image at its original size.

    Uses all ranges for both training and inference, not the known disease
    label. Per-disease masks are color-filter diagnostics, not diagnoses.
    """
    ranges = DISEASE_RANGES if disease_ranges is None else disease_ranges
    hsv = to_hsv(bgr)

    per_disease = {
        name: threshold_mask(hsv, hsv_ranges)
        for name, hsv_ranges in ranges.items()
    }

    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for disease_mask in per_disease.values():
        combined = np.bitwise_or(combined, disease_mask)

    leaf = leaf_region(hsv)
    lesions = clean_mask(cv2.bitwise_and(combined, leaf))
    per_disease = {
        name: cv2.bitwise_and(mask, lesions)
        for name, mask in per_disease.items()
    }

    return SegmentationResult(
        hsv=hsv,
        leaf_mask=leaf,
        lesion_mask=lesions,
        masked=apply_mask(bgr, lesions),
        per_disease=per_disease,
    )

def synthetic_leaf(size: int = 256, seed: int = 0) -> np.ndarray:
    """Create a fake leaf with orange spots for tests without real data."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), (200, 200, 200), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.ellipse(
        image,
        center,
        (size // 3, size // 2 - 20),
        30,
        0,
        360,
        (40, 150, 60),
        -1,
    )

    for _ in range(5):
        dx, dy = rng.integers(-size // 6, size // 6, size=2)
        radius = int(rng.integers(6, 12))
        cv2.circle(
            image,
            (center[0] + int(dx), center[1] + int(dy)),
            radius,
            (20, 140, 240),
            -1,
        )
    return image

