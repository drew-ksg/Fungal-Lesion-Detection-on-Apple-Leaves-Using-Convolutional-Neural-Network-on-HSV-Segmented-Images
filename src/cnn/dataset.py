"""Purpose: Load the color PlantVillage apple images for CNN experiments.
this loader will read the files from the dataset."""

#delays evaluating annotations until it's needed
from __future__ import annotations

import csv 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Callable

import torch #the machine-learning library used by the CNN
from PIL import Image #imports image tool from Pillow, a python image-processing library
from torch.utils.data import Dataset

#the four image-level labels supplied by PlantVillage
IMAGE_CLASSES = (
    "apple_scab",
    "black_rot",
    "cedar_apple_rust",
    "healthy",
)
#dictionary that assigns each class a number, in this case 0-3, for the CNN to use
CLASS_TO_INDEX = {name: index for index, name in enumerate(IMAGE_CLASSES)}

#dictionary that translates the original PlantVillage folder names 
#into simpler labels used by our project
_FOLDER_TO_LABEL = {
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "black_rot",
    "Apple___Cedar_apple_rust": "cedar_apple_rust",
    "Apple___healthy": "healthy",
}
#the loader will accept files ending in
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass(frozen=True) #once an ImageRecord is created, its values cannot be changed
class ImageRecord: 
    """Each IR represents one image from the dataset and its info."""
    #fields stored in ImageRecord
    path: Path #sores the location of the image file
    label_name: str #disease name
    label: int #numeric label for the disease 
    split: str | None = None #training/validation/test split
    leaf_id: str | None = None #ID of the leaf associated with the image

#function that converts db folder name to the project label name using our dictionary
def _label_from_folder(folder_name: str) -> str:
    try:
        return _FOLDER_TO_LABEL[folder_name]
    except KeyError as exc:
        raise ValueError(f"unsupported PlantVillage class folder: {folder_name}") from exc

#function finds the image file on system when the manifest only contains a path to it
#the manifest stores paths relative to the repo root while users may construct the 
#dataset from a diff directory
def _resolve_path(raw_path: str, dataset_dir: Path) -> Path:
    path = Path(raw_path)
    candidates = [
        path,
        dataset_dir / path,
        dataset_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve() #returns the path of img file if it exists

    #final fallback handles a manifest whose path contains the usual
    #data/raw/plantvillage_apple prefix.
    parts = path.parts
    marker = "plantvillage_apple"
    if marker in parts:
        suffix = parts[parts.index(marker) + 1 :]
        candidate = dataset_dir.joinpath(*suffix)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"image listed in manifest was not found: {raw_path!r}. "
        f"Checked relative to {dataset_dir} and the current directory."
    )

def _records_from_manifest(
    manifest: Path,
    dataset_dir: Path,
    split: str | None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if split is not None and row.get("split") != split:
                continue

            raw_label = row.get("label", "")
            label_name = _FOLDER_TO_LABEL.get(raw_label, raw_label.lower())
            if label_name not in CLASS_TO_INDEX:
                raise ValueError(f"unsupported image label in manifest: {raw_label!r}")

            records.append(
                ImageRecord(
                    path=_resolve_path(row["path"], dataset_dir),
                    label_name=label_name,
                    label=CLASS_TO_INDEX[label_name],
                    split=row.get("split") or None,
                    leaf_id=row.get("leaf_id") or None,
                )
            )
    return records

#discover images when a manifest hasn't been generated.
def _records_from_folders(dataset_dir: Path, split: str | None) -> list[ImageRecord]:
    split_dirs = [dataset_dir / split] if split else sorted(
        path for path in dataset_dir.iterdir() if path.is_dir()
    )
    records: list[ImageRecord] = []
    for split_dir in split_dirs:
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label_name = _label_from_folder(class_dir.name)
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    records.append(
                        ImageRecord(
                            path=image_path.resolve(),
                            label_name=label_name,
                            label=CLASS_TO_INDEX[label_name],
                            split=split_dir.name,
                        )
                    )
    return records


#PyTorch dataset for images. each item is a dictionary containing the label and empty box
#fields. the fields make the interface ready for CVAT annotations
class AppleLeafDataset(Dataset):

    def __init__(
        self,
        dataset_dir: str | Path = "data/raw/plantvillage_apple",
        split: str | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"dataset directory does not exist: {self.dataset_dir}")

        manifest = self.dataset_dir / "manifest.csv"
        if manifest.exists():
            self.records = _records_from_manifest(manifest, self.dataset_dir, split)
        else:
            self.records = _records_from_folders(self.dataset_dir, split)

        if not self.records:
            split_text = f" for split {split!r}" if split else ""
            raise ValueError(f"no supported images found in {self.dataset_dir}{split_text}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(record.label, dtype=torch.long),
            "label_name": record.label_name,
            "path": str(record.path),
            "leaf_id": record.leaf_id,
            #these remain empty until CVAT annotations
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "box_labels": torch.zeros((0,), dtype=torch.long),
        }