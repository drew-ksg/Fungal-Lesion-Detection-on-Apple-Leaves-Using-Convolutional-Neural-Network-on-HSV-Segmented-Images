"""Save a four-panel figure of the HSV process for one leaf.

    python -m src.hsv.demo --image path/to/leaf.jpg
    python -m src.hsv.demo --image path/to/leaf.jpg --disease cedar_apple_rust
    python -m src.hsv.demo --synthetic
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .segment import draw_boxes, segment_leaf, synthetic_leaf  # noqa: E402
from .thresholds import DISEASE_RANGES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="leaf photo to segment")
    source.add_argument("--synthetic", action="store_true", help="use a fake test leaf")
    parser.add_argument("--disease", choices=sorted(DISEASE_RANGES),
                        help="use only this disease's thresholds (default: all)")
    parser.add_argument("--out", type=Path, default=Path("outputs/hsv_demo.png"))
    args = parser.parse_args()

    if args.synthetic:
        bgr = synthetic_leaf()
    else:
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            parser.error(f"could not read image: {args.image}")

    ranges = {args.disease: DISEASE_RANGES[args.disease]} if args.disease else None
    result = segment_leaf(bgr, disease_ranges=ranges)
    panels = [
        ("Original", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None),
        ("Binary mask", result.lesion_mask, "gray"),
        ("Masked picture", cv2.cvtColor(result.masked, cv2.COLOR_BGR2RGB), None),
        (f"Candidates ({len(result.boxes)})",
         cv2.cvtColor(draw_boxes(bgr, result.boxes), cv2.COLOR_BGR2RGB), None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}  ({len(result.boxes)} candidate regions)")


if __name__ == "__main__":
    main()
