"""HSV threshold ranges for each disease.

A threshold is a lower and upper limit on Hue, Saturation, and Value. A pixel
is selected only when ALL THREE of its measurements fall inside the range.

OpenCV scales (important when copying values from other sources):
    Hue        0-179   (half of the usual 0-360 degrees)
    Saturation 0-255
    Value      0-255

These are STARTING VALUES, set from pixel measurements on about 60 PlantVillage
apple leaves per class. Tune them by hand (see notebooks/01_hsv_demo.ipynb).

What the measurements showed:
  - Healthy green leaf tissue sits around hue 40-80.
  - The PlantVillage background is gray-purple or warm gray with low
    saturation, so the leaf mask (LEAF_RANGES) is what keeps it out.
  - Rust and black rot are far from green in hue and separate well.
  - Scab is the hard case: its lesions are olive-green, inside the healthy hue
    band, and mainly DARKER. The scab range will also catch shadowed healthy
    tissue; the CNN step has to sort that out.

A disease can have more than one range. Red and purple sit at both ends of the
hue scale (near 0 and near 179), so black rot's purple margin needs two.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HSVRange:
    """One lower/upper limit on (hue, saturation, value)."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        limits = (179, 255, 255)
        for lo, hi, top, name in zip(self.lower, self.upper, limits, "HSV"):
            if not (0 <= lo <= hi <= top):
                raise ValueError(
                    f"{name} range {lo}-{hi} is invalid; must satisfy 0 <= lower <= upper <= {top}"
                )


# Healthy green leaf tissue. Used to find the leaf itself, not lesions.
LEAF_RANGES: list[HSVRange] = [
    HSVRange(lower=(25, 40, 30), upper=(95, 255, 255)),
]

# Suspected lesion colors for each disease.
DISEASE_RANGES: dict[str, list[HSVRange]] = {
    # Yellow-orange spots, brighter and less saturated than healthy green.
    "cedar_apple_rust": [
        HSVRange(lower=(10, 30, 100), upper=(32, 255, 255)),
    ],
    # Dark olive-green to brown patches. Overlaps shadowed healthy tissue.
    "apple_scab": [
        HSVRange(lower=(20, 40, 15), upper=(48, 255, 80)),
    ],
    # Frogeye leaf spot: tan-brown center with a purple margin.
    "black_rot": [
        HSVRange(lower=(5, 40, 30), upper=(22, 255, 220)),     # brown center
        HSVRange(lower=(150, 50, 30), upper=(179, 255, 200)),  # purple-red margin
        HSVRange(lower=(0, 50, 30), upper=(4, 255, 200)),      # red wraps past 0
    ],
}
