"""HSV segmentation: picture -> HSV -> thresholds -> binary mask -> masked picture.

Typical use:

    import cv2
    from src.hsv import segment

    bgr = cv2.imread("leaf.jpg")
    result = segment.segment_leaf(bgr)
    result.masked      # original picture with only suspected lesion pixels kept
    result.boxes       # candidate lesion boxes (x, y, w, h) to send to the CNN

Note: OpenCV loads images in BGR order, not RGB. Every function here expects
BGR, which is what cv2.imread returns.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .thresholds import DISEASE_RANGES, LEAF_RANGES, HSVRange


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Step 2: convert a BGR picture to HSV."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a color image with 3 channels")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def threshold_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    """Step 3: build a binary mask from one or more HSV ranges.

    A pixel turns white (255) when its hue, saturation, AND value all fall
    inside at least one range. Every other pixel is black (0).
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, np.array(r.lower, np.uint8), np.array(r.upper, np.uint8))
    return mask


def clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    """Remove small specks (opening) and fill small holes (closing)."""
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def leaf_region(hsv: np.ndarray, grow: int = 3) -> np.ndarray:
    """Mask of the leaf itself, so background pixels are never called lesions.

    Built from healthy green only: keep the largest green region, fill the
    holes inside it (that is where lesions sit), then grow it by `grow` pixels
    so lesions touching the leaf edge are not cut off.
    """
    leaf = threshold_mask(hsv, LEAF_RANGES)
    leaf = clean_mask(leaf, open_size=5, close_size=15)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(leaf)
    if count <= 1:
        return leaf
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    leaf = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill holes inside the leaf outline (lesions that fell outside both ranges).
    contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(leaf)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    if grow > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        filled = cv2.dilate(filled, kernel)
    return filled


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Step 4: keep only the original pixels where the mask is white."""
    return cv2.bitwise_and(bgr, bgr, mask=mask)


def candidate_boxes(
    mask: np.ndarray, min_area: int = 30, pad: int = 4
) -> list[tuple[int, int, int, int]]:
    """Group white pixels into regions and return a box (x, y, w, h) around each.

    Regions smaller than min_area pixels are ignored. Each box is padded by
    `pad` pixels so the CNN sees a little healthy tissue around the lesion.
    """
    h, w = mask.shape[:2]
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return boxes


@dataclass
class SegmentationResult:
    hsv: np.ndarray
    leaf_mask: np.ndarray
    lesion_mask: np.ndarray            # all diseases combined
    masked: np.ndarray                 # original picture, lesion pixels only
    boxes: list[tuple[int, int, int, int]]
    per_disease: dict[str, np.ndarray] = field(default_factory=dict)


def segment_leaf(
    bgr: np.ndarray,
    disease_ranges: dict[str, list[HSVRange]] | None = None,
    min_area: int = 30,
) -> SegmentationResult:
    """Run the full HSV process on one picture."""
    ranges = DISEASE_RANGES if disease_ranges is None else disease_ranges
    hsv = to_hsv(bgr)

    per_disease = {name: threshold_mask(hsv, rs) for name, rs in ranges.items()}
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for m in per_disease.values():
        combined |= m

    leaf = leaf_region(hsv)
    lesions = clean_mask(cv2.bitwise_and(combined, leaf))
    per_disease = {k: cv2.bitwise_and(v, lesions) for k, v in per_disease.items()}

    return SegmentationResult(
        hsv=hsv,
        leaf_mask=leaf,
        lesion_mask=lesions,
        masked=apply_mask(bgr, lesions),
        boxes=candidate_boxes(lesions, min_area=min_area),
        per_disease=per_disease,
    )


def draw_boxes(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of the picture with boxes drawn on it."""
    out = bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def synthetic_leaf(size: int = 256, seed: int = 0) -> np.ndarray:
    """A fake leaf with orange 'rust' spots, for testing without real data."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (200, 200, 200), np.uint8)   # gray background
    center = (size // 2, size // 2)
    cv2.ellipse(img, center, (size // 3, size // 2 - 20), 30, 0, 360, (40, 150, 60), -1)
    for _ in range(5):
        dx, dy = rng.integers(-size // 6, size // 6, size=2)
        r = int(rng.integers(6, 12))
        cv2.circle(img, (center[0] + int(dx), center[1] + int(dy)), r, (20, 140, 240), -1)
    return img
