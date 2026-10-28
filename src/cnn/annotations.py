"""Purpose: Read lesion boxes exported from CVAT in YOLO format.
This file converts normalized YOLO coordinates into pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# The numbers match the region classifier in src/cnn/model.py.
LESION_CLASSES = (
    "not_lesion",
    "apple_scab",
    "black_rot",
    "cedar_apple_rust",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(LESION_CLASSES)}

# CVAT uses only these three labels for lesion boxes.
CVAT_CLASS_TO_LABEL = {
    0: "apple_scab",
    1: "black_rot",
    2: "cedar_apple_rust",
}

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class LesionAnnotation:
    """Stores one lesion box and its disease label."""

    box: Box
    label_name: str
    label: int


def _check_image_size(image_width: int, image_height: int) -> None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image width and height must be greater than zero")


def _yolo_box_to_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Box:
    """Convert normalized YOLO values into (x, y, width, height) pixels."""
    x = (x_center - width / 2) * image_width
    y = (y_center - height / 2) * image_height
    box_width = width * image_width
    box_height = height * image_height

    # Keep boxes inside the image if an annotation touches an edge.
    right = min(image_width, x + box_width)
    bottom = min(image_height, y + box_height)
    x = max(0.0, x)
    y = max(0.0, y)

    return (x, y, max(0.0, right - x), max(0.0, bottom - y))


def read_yolo_labels(
    label_path: str | Path,
    image_width: int,
    image_height: int,
) -> list[LesionAnnotation]:
    """Read a CVAT YOLO 1.1 label file for one image.

    Each non-empty line must contain:

        class_id x_center y_center width height

    The coordinate values are normalized from 0 to 1 in YOLO format.  The
    returned boxes use pixel coordinates so they can be drawn on the image
    and compared with predictions using IoU.
    """
    _check_image_size(image_width, image_height)
    label_path = Path(label_path)

    # Healthy images have no lesion boxes, so a missing label file is valid.
    if not label_path.exists():
        return []

    annotations: list[LesionAnnotation] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number} must contain 5 values, found {len(parts)}"
            )

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{label_path}:{line_number} contains a non-numeric value"
            ) from exc

        if class_id not in CVAT_CLASS_TO_LABEL:
            raise ValueError(f"unsupported CVAT class id {class_id} in {label_path}")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"YOLO coordinates must be between 0 and 1 in {label_path}")
        if values[2] == 0 or values[3] == 0:
            raise ValueError(f"YOLO box width and height must be greater than zero in {label_path}")

        label_name = CVAT_CLASS_TO_LABEL[class_id]
        annotations.append(
            LesionAnnotation(
                box=_yolo_box_to_pixels(*values, image_width, image_height),
                label_name=label_name,
                label=CLASS_TO_INDEX[label_name],
            )
        )

    return annotations
