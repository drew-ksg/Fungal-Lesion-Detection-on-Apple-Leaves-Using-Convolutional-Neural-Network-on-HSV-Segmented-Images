# Data

Datasets are **not committed** to Git. Each team member downloads them locally.

| Folder | Contents | Committed? |
| --- | --- | --- |
| `raw/` | Original images from `scripts/download_plantvillage.py` | No |
| `processed/` | Resized images, crops, and masks made by our code | No |
| `labels/` | Our hand-drawn lesion boxes (exported from CVAT as YOLO 1.1) | **Yes** |

Lesion labels are small text files, so they are committed and shared through
GitHub. Keep the export format consistent (see `annotation/LABELING_GUIDE.md`).

## Source

PlantVillage, apple classes only, downloaded from
<https://github.com/spMohanty/PlantVillage-Dataset> (same images as
<https://huggingface.co/datasets/mohanty/PlantVillage>; CC BY-SA 3.0).

Run `python scripts/download_plantvillage.py` from the repository root.
