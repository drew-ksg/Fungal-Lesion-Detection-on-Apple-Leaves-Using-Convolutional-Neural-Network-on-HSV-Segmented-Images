"""Tests for the image loader and CVAT annotation reader."""

from pathlib import Path

import pytest
from PIL import Image

from src.cnn.annotations import read_yolo_labels
from src.cnn.dataset import AppleLeafDataset


def _make_image(path: Path) -> None:
    """Create a tiny image so the tests do not need the real dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color=(40, 150, 60)).save(path)


def test_dataset_finds_color_images_without_manifest(tmp_path):
    image_path = tmp_path / "train" / "Apple___Apple_scab" / "leaf.jpg"
    _make_image(image_path)

    dataset = AppleLeafDataset(tmp_path, split="train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["label_name"] == "apple_scab"
    assert sample["label"].item() == 0
    assert sample["image"].size == (20, 10)
    assert sample["boxes"].shape == (0, 4)


def test_dataset_uses_manifest_and_split(tmp_path):
    image_path = tmp_path / "val" / "Apple___healthy" / "healthy.png"
    _make_image(image_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,split,label,leaf_id\n"
        f"{image_path},val,Apple___healthy,leaf-1\n",
        encoding="utf-8",
    )

    dataset = AppleLeafDataset(tmp_path, split="val")

    assert len(dataset) == 1
    assert dataset[0]["label_name"] == "healthy"
    assert dataset[0]["leaf_id"] == "leaf-1"


def test_yolo_annotations_convert_to_pixel_boxes(tmp_path):
    labels = tmp_path / "leaf.txt"
    labels.write_text("0 0.5 0.5 0.4 0.6\n", encoding="utf-8")

    annotations = read_yolo_labels(labels, image_width=100, image_height=50)

    assert len(annotations) == 1
    assert annotations[0].label_name == "apple_scab"
    assert annotations[0].label == 1
    assert annotations[0].box == pytest.approx((30.0, 10.0, 40.0, 30.0))


def test_missing_yolo_file_means_no_boxes(tmp_path):
    annotations = read_yolo_labels(tmp_path / "healthy.txt", 100, 50)
    assert annotations == []


def test_bad_yolo_line_is_rejected(tmp_path):
    labels = tmp_path / "bad.txt"
    labels.write_text("0 0.5 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_yolo_labels(labels, 100, 50)
