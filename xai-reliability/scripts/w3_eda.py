"""Week 3-4: the limited EDA the IPR commits to.

Checks only what could invalidate the benchmark: that the subset really is one
image per class, and that the images are large enough that a 224x224 centre
crop is not upsampling. Anything more would be padding - this is a
benchmarking study, not a data-analysis one.
"""
import argparse

import pandas as pd
from PIL import Image

import _bootstrap  # noqa: F401
from src.config import load_config
from src.utils import append_rows, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    table = pd.read_csv(cfg.resolve("data.manifest"))
    if args.limit:
        table = table.iloc[: args.limit]

    sizes, modes = [], []
    for path in table["path"]:
        with Image.open(path) as im:
            sizes.append(im.size)
            modes.append(im.mode)
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    short_side = [min(s) for s in sizes]

    report = {
        "n_images": len(table),
        "n_classes": int(table.class_idx.nunique()),
        "images_per_class_min": int(table.groupby("class_idx").size().min()),
        "images_per_class_max": int(table.groupby("class_idx").size().max()),
        "width_mean": sum(widths) / len(widths),
        "height_mean": sum(heights) / len(heights),
        "short_side_min": min(short_side),
        "n_below_256_short_side": sum(s < 256 for s in short_side),
        "n_non_rgb": sum(m != "RGB" for m in modes),
    }

    out = ensure_dir(cfg.resolve("paths.results")) / "eda_summary.csv"
    append_rows(out, [report])
    for key, value in report.items():
        print(f"{key:28s} {value}")
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
