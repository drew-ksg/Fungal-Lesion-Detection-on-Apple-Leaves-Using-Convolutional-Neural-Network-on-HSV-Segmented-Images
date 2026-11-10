"""Convert a leaf picture into HSV masks and a masked color image.
made this so that the functions in this file can be imported without opening a window or
depending on a specific path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2 #image processing tools from OpenCV
import numpy as np #used to work with images as arrays

#imports the starting HSV ranges from thresholds and the class used to store them
from .thresholds import DISEASE_RANGES, LEAF_RANGES, HSVRange

def to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image into HSV."""
    #checks that the image has 3 color channels instead of being grayscale
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a color image with 3 channels")
    #OpenCV loads colors as BGR, so we convert from BGR to HSV
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def threshold_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    """Create a mask where pixels fall inside at least one HSV range."""
    #starts with a black mask the same height and width as the image
    #uint8 stores whole numbers from 0-255
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in ranges:
        #pixels inside all three HSV limits become white, everything else black
        range_mask = np.asarray(
            cv2.inRange(
                hsv,
                np.array(hsv_range.lower, dtype=np.uint8),
                np.array(hsv_range.upper, dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        #keeps pixels selected by either the existing mask or the new range
        mask = np.bitwise_or(mask, range_mask)
    return mask


def clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    """Remove small specks and fill small holes in a binary mask."""
    if open_size > 1:
        #creates a small ellipse-shaped filter for cleaning the mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        #opening removes small white specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        #closing fills small black gaps inside white regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def leaf_region(hsv: np.ndarray, grow: int = 3) -> np.ndarray:
    """Create a mask for the largest green leaf region in the image."""
    #finds green tissue, then cleans the mask
    leaf = threshold_mask(hsv, LEAF_RANGES)
    leaf = clean_mask(leaf, open_size=5, close_size=15)

    #finds separate white regions and measures their sizes
    count, labels, stats, _ = cv2.connectedComponentsWithStats(leaf)

    #the count includes the background, so 1 means no leaf region was found
    if count <= 1:
        return leaf

    #skips the background and keeps the largest region as the main leaf
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = np.where(labels == largest, 255, 0).astype(np.uint8)

    #finds the outside edges of the selected leaf region
    contours, _ = cv2.findContours(
        leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    #fills the outline so enclosed lesions aren't excluded for being non-green
    filled = np.zeros_like(leaf)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    if grow > 0:
        #expands the mask slightly to include pixels near the leaf edge
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)
        )
        filled = cv2.dilate(filled, kernel)
    return filled


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep the original pixels where the mask is white."""
    #keeps selected pixels in their original colors and makes the rest black
    return cv2.bitwise_and(bgr, bgr, mask=mask)

@dataclass
class SegmentationResult:
    """Stores the results produced by the full HSV process."""

    hsv: np.ndarray #image converted to HSV
    leaf_mask: np.ndarray #estimated area of the leaf
    lesion_mask: np.ndarray #combined mask of possible lesion pixels
    masked: np.ndarray #original color image with excluded pixels made black
    #separate color masks for inspection, not confirmed disease labels
    per_disease: dict[str, np.ndarray] = field(default_factory=dict)


def segment_leaf(
    bgr: np.ndarray,
    disease_ranges: dict[str, list[HSVRange]] | None = None,
) -> SegmentationResult:
    """Combines HSV ranges and return a masked BGR image at its original size.

    Uses all ranges for both training and inference, not the known disease
    label. Per-disease masks are color-filter diagnostics, not diagnoses.
    """
    #uses the default ranges unless custom ranges were provided
    ranges = DISEASE_RANGES if disease_ranges is None else disease_ranges
    hsv = to_hsv(bgr)

    #creates a separate mask for each disease's color ranges
    per_disease = {
        name: threshold_mask(hsv, hsv_ranges)
        for name, hsv_ranges in ranges.items()
    }

    #combines all masks so preprocessing doesn't depend on knowing the disease
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for disease_mask in per_disease.values():
        combined = np.bitwise_or(combined, disease_mask)

    #limits selected pixels to the estimated leaf area, then cleans the mask
    leaf = leaf_region(hsv)
    lesions = clean_mask(cv2.bitwise_and(combined, leaf))

    #limits the diagnostic masks to pixels kept in the cleaned combined mask
    per_disease = {
        name: cv2.bitwise_and(mask, lesions)
        for name, mask in per_disease.items()
    }

    #returns the masks and masked image. the CNN will handle predicting boxes
    return SegmentationResult(
        hsv=hsv,
        leaf_mask=leaf,
        lesion_mask=lesions,
        masked=apply_mask(bgr, lesions),
        per_disease=per_disease,
    )

def synthetic_leaf(size: int = 256, seed: int = 0) -> np.ndarray:
    """Create a fake leaf with orange spots for tests without real data."""
    #the same seed produces the same test image each time
    rng = np.random.default_rng(seed)
    #creates a gray background
    image = np.full((size, size, 3), (200, 200, 200), dtype=np.uint8)
    center = (size // 2, size // 2)
    #draws a filled green ellipse to represent the leaf
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
    #draws five orange spots near the center with random positions and sizes
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

