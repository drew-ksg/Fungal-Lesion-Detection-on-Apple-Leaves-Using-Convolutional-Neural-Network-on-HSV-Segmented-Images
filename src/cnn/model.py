"""Region classifier: decides whether a candidate region is a lesion, and which disease.

The same model is used for both methods:
  - Proposed: regions come from HSV candidate boxes.
  - Baseline: the model is trained and tested on the plain photos, no HSV step.

Classes (index order matters; keep it fixed):
    0 not_lesion   healthy tissue, veins, background, glare
    1 apple_scab
    2 black_rot
    3 cedar_apple_rust
"""

import torch
from torch import nn
from torchvision import models

CLASSES = ["not_lesion", "apple_scab", "black_rot", "cedar_apple_rust"]
INPUT_SIZE = 128  # region crops are resized to INPUT_SIZE x INPUT_SIZE


def build_region_classifier(
    arch: str = "mobilenet_v3_small",
    num_classes: int = len(CLASSES),
    pretrained: bool = True,
) -> nn.Module:
    """Return an ImageNet-pretrained CNN with a new final layer for our classes.

    arch: "mobilenet_v3_small" (fast, small) or "resnet18" (a common baseline).
    pretrained=False skips the weight download, useful for offline tests.
    """
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return net


def device() -> torch.device:
    """Use the GPU when one is available (Colab, CSUB lab machine)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
