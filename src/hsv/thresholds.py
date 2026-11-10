"""this file stores the HSV ranges used to find possible lesion pixels.
these are only starting ranges and will need to be tested on the real dataset later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HSVRange:
    """Stores the lower and upper HSV limits for one color range."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        limits = (179, 255, 255)
        for low, high, top, name in zip(self.lower, self.upper, limits, "HSV"):
            if not (0 <= low <= high <= top):
                raise ValueError(
                    f"{name} range {low}-{high} is invalid; "
                    f"must satisfy 0 <= lower <= upper <= {top}"
                )


# healthy green leaf tissue. this is used to find the leaf itself.
LEAF_RANGES = [
    HSVRange(lower=(25, 40, 30), upper=(95, 255, 255)),
]

# suspected lesion colors for each disease.
DISEASE_RANGES = {
    "cedar_apple_rust": [
        HSVRange(lower=(10, 30, 100), upper=(32, 255, 255)),
    ],
    "apple_scab": [
        HSVRange(lower=(20, 40, 15), upper=(48, 255, 80)),
    ],
    "black_rot": [
        HSVRange(lower=(5, 40, 30), upper=(22, 255, 220)),
        HSVRange(lower=(150, 50, 30), upper=(179, 255, 200)),
        HSVRange(lower=(0, 50, 30), upper=(4, 255, 200)),
    ],
}

