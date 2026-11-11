"""Week 3: build the fixed evaluation subset.

    python scripts/w3_build_subset.py --source imagenetv2 --root D:/data/imagenetv2-matched-frequency-format-val
    python scripts/w3_build_subset.py --source imagenet_val --root D:/data/ILSVRC2012_val_folders

Writes data/subset_manifest.csv. Commit that file to Git: it is the single
definition of what "the dataset" means in this project.
"""
import argparse

import _bootstrap  # noqa: F401
from src.config import load_config
from src.data import build_manifest
from src.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, choices=["imagenet_val", "indexfolder", "imagenetv2", "imagefolder"])
    ap.add_argument("--root", required=True, help="dataset root directory")
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--labels-file", default=None, help="devkit ground truth, for a flat val folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default=None,
                    help="manifest whose images must not be reused (for the IDSDS fine-tune split)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    df = build_manifest(
        source=args.source or cfg.get_path("data.source"),
        root=args.root,
        out_csv=args.out or cfg.resolve("data.manifest"),
        per_class=args.per_class or cfg.get_path("data.per_class", 1),
        seed=cfg["seed"],
        labels_file=args.labels_file,
        exclude_manifest=args.exclude,
    )
    print(df.head())
    print(f"\n{len(df)} images across {df.class_idx.nunique()} classes")


if __name__ == "__main__":
    main()
