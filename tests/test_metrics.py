import pytest

from src.eval.metrics import iou, match_boxes


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_iou_half_overlap():
    # Overlap 50, union 150.
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_match_counts_found_missed_false_alarms():
    labeled = [(0, 0, 10, 10), (50, 50, 10, 10)]
    predicted = [(1, 1, 10, 10), (100, 100, 5, 5)]
    r = match_boxes(predicted, labeled)
    assert (r.true_positives, r.false_positives, r.false_negatives) == (1, 1, 1)
    assert r.precision == 0.5 and r.recall == 0.5


def test_one_lesion_matched_only_once():
    labeled = [(0, 0, 10, 10)]
    predicted = [(0, 0, 10, 10), (1, 0, 10, 10)]
    r = match_boxes(predicted, labeled, scores=[0.9, 0.8])
    assert r.true_positives == 1 and r.false_positives == 1
