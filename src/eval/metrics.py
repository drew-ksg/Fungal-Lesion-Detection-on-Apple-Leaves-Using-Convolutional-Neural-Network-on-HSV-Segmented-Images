"""Scoring predicted lesion boxes against hand-labeled lesion boxes.

Boxes are (x, y, w, h) in pixels.

A predicted box is a correct find (true positive) when it overlaps an
unmatched labeled lesion with IoU >= iou_threshold. Unmatched predictions are
false alarms; unmatched labeled lesions are missed.
"""

from dataclasses import dataclass

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    """Intersection over Union: overlap area / combined area (0 to 1)."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    true_positives: int    # correctly located lesions
    false_positives: int   # false alarms
    false_negatives: int   # missed lesions

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_boxes(
    predicted: list[Box],
    labeled: list[Box],
    scores: list[float] | None = None,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedy one-to-one matching, highest-confidence predictions first."""
    order = list(range(len(predicted)))
    if scores is not None:
        order.sort(key=lambda i: scores[i], reverse=True)

    used = set()
    tp = 0
    for i in order:
        best_j, best_iou = None, iou_threshold
        for j, gt in enumerate(labeled):
            if j in used:
                continue
            v = iou(predicted[i], gt)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j is not None:
            used.add(best_j)
            tp += 1

    return MatchResult(
        true_positives=tp,
        false_positives=len(predicted) - tp,
        false_negatives=len(labeled) - tp,
    )
