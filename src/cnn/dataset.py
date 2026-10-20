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
