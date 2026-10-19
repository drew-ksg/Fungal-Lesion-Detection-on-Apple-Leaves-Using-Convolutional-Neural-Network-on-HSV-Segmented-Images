"""Download the apple classes of PlantVillage and split them into train/val/test.

    python scripts/download_plantvillage.py                  # all apple images (~52 MB)
    python scripts/download_plantvillage.py --limit 20       # 20 per class, quick test
    python scripts/download_plantvillage.py --config segmented

Source: the original PlantVillage GitHub repository (spMohanty/PlantVillage-Dataset),
the same images as the Hugging Face copy (mohanty/PlantVillage). We use GitHub
because Git can fetch only the four apple folders, while the Hugging Face copy
runs a loader script that downloads the full 2 GB dataset and does not work
with `datasets` 4.0 or newer.

Requires `git` to be installed.

Output:
    data/raw/plantvillage_apple/<split>/<class>/<image>.jpg
    data/raw/plantvillage_apple/manifest.csv   (path, split, label, leaf_id)

Several photos can show the same physical leaf. Splits are made by leaf_id, so
one leaf never appears in both training and test sets.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
APPLE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch_repo(cache: Path, config: str) -> Path:
    """Sparse-clone only the apple folders we need (plus the leaf map)."""
    if shutil.which("git") is None:
        sys.exit("git is not installed. Install it from https://git-scm.com and try again.")
    if not (cache / ".git").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning PlantVillage (metadata only) into {cache} ...")
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(cache)])
    paths = ["/leaf-map.json"] + [f"/raw/{config}/{c}/" for c in APPLE_CLASSES]
    print("Downloading apple images ...")
    run(["git", "sparse-checkout", "set", "--no-cone", *paths], cwd=cache)
    return cache


def leaf_id_for(file_name: str, class_name: str, leaf_map: dict) -> str:
    """Same leaf-grouping rule as the official PlantVillage loader."""
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()

    suggestions = leaf_map.get(ident.lower().strip())
    if suggestions:
        if len(suggestions) == 1:
            return suggestions[0]
        for s in suggestions:
            if class_name in s:
                return s
    return f"fallback_{ident}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="color", choices=["color", "grayscale", "segmented"])
    parser.add_argument("--limit", type=int, default=None, help="max images per class")
    parser.add_argument("--val", type=float, default=0.15, help="share of leaves for validation")
    parser.add_argument("--test", type=float, default=0.15, help="share of leaves for test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/raw/plantvillage_apple"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/_plantvillage_repo"))
    args = parser.parse_args()

    repo = fetch_repo(args.cache, args.config)
    leaf_map_path = repo / "leaf-map.json"
    leaf_map = json.loads(leaf_map_path.read_text()) if leaf_map_path.exists() else {}

    rng = random.Random(args.seed)
    rows = []
    for cls in APPLE_CLASSES:
        src_dir = repo / "raw" / args.config / cls
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.limit:
            files = files[: args.limit]

        # Group photos by physical leaf, then split the leaves.
        by_leaf: dict[str, list[Path]] = defaultdict(list)
        for p in files:
            by_leaf[leaf_id_for(p.name, cls, leaf_map)].append(p)
        leaves = sorted(by_leaf)
        rng.shuffle(leaves)
        n_test = round(len(leaves) * args.test)
        n_val = round(len(leaves) * args.val)
        split_of = {lf: "test" for lf in leaves[:n_test]}
        split_of.update({lf: "val" for lf in leaves[n_test:n_test + n_val]})
        split_of.update({lf: "train" for lf in leaves[n_test + n_val:]})

        counts = defaultdict(int)
        for leaf, photos in by_leaf.items():
            split = split_of[leaf]
            dest_dir = args.out / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for p in photos:
                dest = dest_dir / p.name.replace(" ", "_")
                shutil.copy2(p, dest)
                rows.append({"path": dest.as_posix(), "split": split, "label": cls, "leaf_id": leaf})
                counts[split] += 1
        print(f"{cls:<26} train {counts['train']:>4}  val {counts['val']:>4}  test {counts['test']:>4}"
              f"  ({len(leaves)} leaves)")

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "split", "label", "leaf_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest} ({len(rows)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
