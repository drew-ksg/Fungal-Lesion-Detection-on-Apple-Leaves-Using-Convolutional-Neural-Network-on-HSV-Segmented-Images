"""Check that the development environment is set up.

    python scripts/check_env.py

Prints Python and package versions, whether a GPU is available, and runs the
HSV pipeline on a synthetic leaf as a quick smoke test.
"""

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("PIL", "pillow"),
    ("sklearn", "scikit-learn"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
]


def main() -> int:
    print(f"Python    {platform.python_version()}  ({platform.system()} {platform.machine()})")
    ok = sys.version_info >= (3, 10)
    if not ok:
        print("  !! Python 3.10 or newer is required")

    missing = []
    for module, pip_name in PACKAGES:
        try:
            mod = importlib.import_module(module)
            print(f"{pip_name:<14}{getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{pip_name:<14}MISSING")
            missing.append(pip_name)

    try:
        import torch

        if torch.cuda.is_available():
            print(f"GPU       CUDA: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("GPU       Apple Metal (MPS)")
        else:
            print("GPU       none (CPU only; use Colab for training)")
    except ImportError:
        pass

    try:
        from src.hsv.segment import segment_leaf, synthetic_leaf

        n = len(segment_leaf(synthetic_leaf()).boxes)
        print(f"HSV test  OK, found {n} candidate regions on the synthetic leaf")
    except Exception as e:  # noqa: BLE001
        print(f"HSV test  FAILED: {e}")
        ok = False

    if missing:
        print("\nInstall missing packages with:  pip install -r requirements.txt")
        ok = False
    print("\nEnvironment ready." if ok else "\nEnvironment has problems (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())