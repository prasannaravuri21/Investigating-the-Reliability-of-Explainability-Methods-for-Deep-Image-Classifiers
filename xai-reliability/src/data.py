"""Week 3: build and load the fixed evaluation subset.

The whole study depends on one fixed list of images. That list is written once
to a CSV manifest (path, class index, class name, source) and every later stage
reads it, so a result can never silently depend on a different sample.

Three sources are supported, all of which produce the same manifest format:

  imagenet_val  the official ILSVRC2012 validation set, either as
                <root>/<wnid>/<image>.JPEG (class-folder layout) or as a flat
                folder plus a ground-truth text file.
  indexfolder   <root>/<class_idx>/ - folder names are ImageNet class indices.
                This is what scripts/w3_prepare_imagenet_val.py writes, and it
                is also the native layout of ImageNetV2 matched-frequency.
                `imagenetv2` is accepted as an alias.
  imagefolder   any torchvision ImageFolder-style directory (used for
                Imagenette while developing the pipeline).

Switching source is a one-line change, which is deliberate: development can
start on ImageNetV2 and move to the official validation set without touching
any experiment code.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import ensure_dir, get_logger

LOG = get_logger()
IMG_EXTENSIONS = {".jpeg", ".jpg", ".png", ".ppm", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Class-index conventions
# --------------------------------------------------------------------------- #
def imagenet_categories() -> list[str]:
    """Human-readable class names in torchvision's class-index order."""
    try:
        from torchvision.models import ResNet50_Weights

        return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    except Exception:  # torchvision not installed yet: fall back to indices
        LOG.warning("torchvision unavailable; class_name column will hold indices")
        return [str(i) for i in range(1000)]


def wnid_to_index(root: Path) -> dict[str, int]:
    """Map WordNet ids to class indices.

    torchvision's pretrained models use the standard ILSVRC2012 ordering, which
    is the alphabetical order of the WordNet ids. A class-folder validation
    directory therefore gives the correct mapping by sorting its folder names.
    """
    wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(wnids) != 1000:
        LOG.warning("expected 1000 class folders under %s, found %d", root, len(wnids))
    return {wnid: i for i, wnid in enumerate(wnids)}


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def _collect_class_folders(root: Path, label_from_folder) -> dict[int, list[Path]]:
    by_class: dict[int, list[Path]] = {}
    for folder in sorted(d for d in root.iterdir() if d.is_dir()):
        label = label_from_folder(folder.name)
        if label is None:
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
        if files:
            by_class.setdefault(label, []).extend(files)
    return by_class


def build_manifest(
    source: str,
    root: str | Path,
    out_csv: str | Path,
    per_class: int = 1,
    seed: int = 0,
    labels_file: str | Path | None = None,
    exclude_manifest: str | Path | None = None,
) -> pd.DataFrame:
    """Sample `per_class` images per class and write the manifest CSV.

    `exclude_manifest` drops every image already used by another manifest. That
    is how the IDSDS fine-tuning split is kept disjoint from the evaluation
    subset: fine-tuning on an image and then measuring faithfulness on the same
    image would make the deletion scores meaningless.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    rng = random.Random(seed)
    categories = imagenet_categories()

    if source == "imagenet_val":
        flat = not any(d.is_dir() for d in root.iterdir())
        if flat:
            if labels_file is None:
                raise ValueError(
                    "a flat validation folder needs --labels-file "
                    "(ILSVRC2012_validation_ground_truth.txt from the devkit)"
                )
            by_class = _flat_val_index(root, Path(labels_file))
        else:
            mapping = wnid_to_index(root)
            by_class = _collect_class_folders(root, lambda name: mapping.get(name))
    elif source in {"indexfolder", "imagenetv2"}:
        # Folder names are the ImageNet class indices already.
        by_class = _collect_class_folders(root, lambda name: int(name) if name.isdigit() else None)
    elif source == "imagefolder":
        folders = sorted(d.name for d in root.iterdir() if d.is_dir())
        mapping = {name: i for i, name in enumerate(folders)}
        by_class = _collect_class_folders(root, lambda name: mapping.get(name))
    else:
        raise ValueError(f"unknown source {source!r}")

    excluded: set[str] = set()
    if exclude_manifest is not None and Path(exclude_manifest).exists():
        prior = pd.read_csv(exclude_manifest)
        excluded = {str(Path(p).resolve()) for p in prior["path"]}
        LOG.info("excluding %d images used by %s", len(excluded), exclude_manifest)

    rows = []
    for class_idx in sorted(by_class):
        files = [f for f in sorted(by_class[class_idx]) if str(f.resolve()) not in excluded]
        if not files:
            continue
        chosen = files if len(files) <= per_class else rng.sample(files, per_class)
        for path in sorted(chosen):
            rows.append(
                {
                    "class_idx": class_idx,
                    "class_name": categories[class_idx] if class_idx < len(categories) else str(class_idx),
                    "path": str(path.resolve()),
                    "source": source,
                }
            )

    df = pd.DataFrame(rows).sort_values("class_idx").reset_index(drop=True)
    df.insert(0, "image_id", df.index)
    out = Path(out_csv)
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    LOG.info("manifest: %d images, %d classes -> %s", len(df), df.class_idx.nunique(), out)
    return df


def _flat_val_index(root: Path, labels_file: Path) -> dict[int, list[Path]]:
    """Index a flat ILSVRC2012 validation folder using the devkit labels.

    The devkit ground truth is 1-based and in ILSVRC2012 *ID* order, which is
    not the same as the alphabetical WordNet-id order the models use. The
    mapping file `data/ilsvrc2012_val_labels.csv` (image_name,class_idx) is the
    safer route; if it exists it is used in preference to the raw devkit file.
    """
    override = root.parent / "ilsvrc2012_val_labels.csv"
    if override.exists():
        LOG.info("using label mapping %s", override)
        table = pd.read_csv(override)
        by_class: dict[int, list[Path]] = {}
        for name, idx in zip(table.image_name, table.class_idx):
            p = root / str(name)
            if p.exists():
                by_class.setdefault(int(idx), []).append(p)
        return by_class

    raise FileNotFoundError(
        "A flat validation folder needs a verified name -> class-index mapping.\n"
        "Write data/ilsvrc2012_val_labels.csv with columns image_name,class_idx "
        "(0-based, torchvision ordering), or re-arrange the images into "
        "<root>/<wnid>/ folders, which is unambiguous."
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SubsetDataset(Dataset):
    """Reads the manifest and applies a model-specific transform.

    Returns (tensor, class_idx, image_id) so every record written downstream can
    be joined back to the manifest.
    """

    def __init__(self, manifest: str | Path, transform=None, limit: int | None = None):
        self.table = pd.read_csv(manifest)
        if limit is not None:
            self.table = self.table.iloc[:limit].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int):
        row = self.table.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["class_idx"]), int(row["image_id"])


def make_loader(manifest, transform, batch_size: int, num_workers: int = 4, limit=None, shuffle=False):
    from torch.utils.data import DataLoader

    ds = SubsetDataset(manifest, transform=transform, limit=limit)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
