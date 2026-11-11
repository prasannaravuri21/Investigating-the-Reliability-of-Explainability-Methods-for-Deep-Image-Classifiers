"""Week 3: turn any ILSVRC2012 validation download into one canonical layout.

    <out>/<class_idx>/<name>.JPEG      class_idx is 0-999, torchvision ordering

All three realistic download routes are handled, because they hand you three
different formats and only the last one carries no labels at all:

  --mode hf       Hugging Face `imagenet-1k` or `timm/imagenet-1k-wds` through
                  the `datasets` library. Labels come with the data. Needs
                  `pip install datasets` and `hf auth login` after accepting
                  the ImageNet terms on the dataset page. Easiest route.

  --mode wds      Already-downloaded webdataset shards
                  (imagenet1k-validation-*.tar). Each sample is a .jpg plus a
                  .cls holding the integer class index.

  --mode devkit   The raw ILSVRC2012_img_val.tar from image-net.org plus the
                  devkit. This is the only route where the labels need real
                  work, and getting it wrong is silent: the devkit ground truth
                  is 1-based in ILSVRC2012_ID order, which is NOT the
                  alphabetical-WNID order that pretrained models use. meta.mat
                  supplies ILSVRC2012_ID -> WNID; sorting the WNIDs then gives
                  the model's class index. Both steps are done here rather than
                  left to a shell script.

Afterwards:

    python scripts/w3_build_subset.py --source indexfolder --root <out>

Verify with w4_baseline_accuracy.py before anything else. If ResNet-50 does not
land near 80% top-1, the label mapping is wrong, and that is exactly the failure
this script exists to prevent.
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401
from src.utils import ensure_dir, get_logger

LOG = get_logger()


def _dest(out: Path, class_idx: int, name: str) -> Path:
    folder = ensure_dir(out / str(class_idx))
    return folder / name


# --------------------------------------------------------------------------- #
# Route 1: the datasets library
# --------------------------------------------------------------------------- #
def from_hf(repo: str, out: Path, split: str = "validation", limit: int | None = None) -> int:
    from datasets import load_dataset

    LOG.info("streaming %s [%s] - the first call downloads the validation shards", repo, split)
    ds = load_dataset(repo, split=split)
    written = 0
    for i, record in enumerate(ds):
        if limit and written >= limit:
            break
        label = int(record["label"] if "label" in record else record["cls"])
        image = record["image"] if "image" in record else record["jpg"]
        path = _dest(out, label, f"val_{i:08d}.JPEG")
        if not path.exists():
            image.convert("RGB").save(path, quality=95)
        written += 1
        if written % 5000 == 0:
            LOG.info("  %d images written", written)
    return written


# --------------------------------------------------------------------------- #
# Route 2: webdataset shards
# --------------------------------------------------------------------------- #
def from_wds(shard_dir: Path, out: Path, limit: int | None = None) -> int:
    shards = sorted(shard_dir.glob("*validation*.tar"))
    if not shards:
        raise FileNotFoundError(f"no *validation*.tar shards under {shard_dir}")
    LOG.info("%d shards found", len(shards))

    written = 0
    for shard in shards:
        pending: dict[str, dict] = {}
        with tarfile.open(shard) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key, _, ext = member.name.partition(".")
                entry = pending.setdefault(key, {})
                payload = tar.extractfile(member)
                if payload is None:
                    continue
                if ext.lower() in {"jpg", "jpeg", "png"}:
                    entry["image"] = payload.read()
                elif ext.lower() in {"cls", "txt"}:
                    entry["label"] = int(payload.read().decode().strip())

                if "image" in entry and "label" in entry:
                    path = _dest(out, entry["label"], f"{key}.JPEG")
                    if not path.exists():
                        path.write_bytes(entry["image"])
                    written += 1
                    pending.pop(key)
                    if limit and written >= limit:
                        return written
        LOG.info("  %s done (%d total)", shard.name, written)
    return written


# --------------------------------------------------------------------------- #
# Route 3: the raw tar plus the devkit
# --------------------------------------------------------------------------- #
def devkit_index_map(meta_mat: Path, ground_truth: Path) -> list[int]:
    """Per-image class indices in torchvision ordering, image 1 first.

    Two hops, both easy to get wrong:
      devkit label (1-based ILSVRC2012_ID)  ->  WNID        [meta.mat]
      WNID                                 ->  class index [sorted WNIDs]
    """
    from scipy.io import loadmat

    meta = loadmat(str(meta_mat), squeeze_me=True)["synsets"]
    # Only the 1000 leaf synsets are ILSVRC2012 classes.
    id_to_wnid = {int(row["ILSVRC2012_ID"]): str(row["WNID"]) for row in meta if int(row["ILSVRC2012_ID"]) <= 1000}
    wnid_to_idx = {wnid: i for i, wnid in enumerate(sorted(id_to_wnid.values()))}
    if len(wnid_to_idx) != 1000:
        raise ValueError(f"expected 1000 synsets in meta.mat, got {len(wnid_to_idx)}")

    with open(ground_truth, "r", encoding="utf-8") as fh:
        devkit_ids = [int(line.strip()) for line in fh if line.strip()]
    LOG.info("devkit ground truth: %d entries", len(devkit_ids))
    return [wnid_to_idx[id_to_wnid[i]] for i in devkit_ids]


def from_devkit(val_tar: Path, meta_mat: Path, ground_truth: Path, out: Path,
                limit: int | None = None) -> int:
    indices = devkit_index_map(meta_mat, ground_truth)
    written = 0
    with tarfile.open(val_tar) as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith((".jpeg", ".jpg")):
                continue
            # Files are named ILSVRC2012_val_00000001.JPEG; the number is the
            # 1-based index into the ground-truth list.
            stem = Path(member.name).stem
            try:
                number = int(stem.split("_")[-1])
            except ValueError:
                LOG.warning("skipping unexpected name %s", member.name)
                continue
            payload = tar.extractfile(member)
            if payload is None:
                continue
            path = _dest(out, indices[number - 1], f"{stem}.JPEG")
            if not path.exists():
                path.write_bytes(payload.read())
            written += 1
            if limit and written >= limit:
                break
            if written % 5000 == 0:
                LOG.info("  %d images written", written)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["hf", "wds", "devkit"])
    ap.add_argument("--out", required=True, help="output root; becomes <out>/<class_idx>/")
    ap.add_argument("--repo", default="imagenet-1k", help="hf mode: dataset repo id")
    ap.add_argument("--split", default="validation", help="hf mode: split name")
    ap.add_argument("--shard-dir", default=None, help="wds mode: folder holding the validation tars")
    ap.add_argument("--val-tar", default=None, help="devkit mode: ILSVRC2012_img_val.tar")
    ap.add_argument("--meta-mat", default=None, help="devkit mode: meta.mat from the devkit")
    ap.add_argument("--ground-truth", default=None,
                    help="devkit mode: ILSVRC2012_validation_ground_truth.txt")
    ap.add_argument("--limit", type=int, default=None, help="stop early, for a dry run")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    if args.mode == "hf":
        written = from_hf(args.repo, out, args.split, args.limit)
    elif args.mode == "wds":
        if not args.shard_dir:
            ap.error("--shard-dir is required for --mode wds")
        written = from_wds(Path(args.shard_dir), out, args.limit)
    else:
        missing = [f for f in ("val_tar", "meta_mat", "ground_truth") if not getattr(args, f)]
        if missing:
            ap.error(f"--mode devkit needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
        written = from_devkit(Path(args.val_tar), Path(args.meta_mat),
                              Path(args.ground_truth), out, args.limit)

    classes = sum(1 for d in out.iterdir() if d.is_dir())
    LOG.info("%d images across %d class folders -> %s", written, classes, out)
    if classes != 1000 and not args.limit:
        LOG.warning("expected 1000 class folders, found %d", classes)
    print("\nNext:")
    print(f"  python scripts/w3_build_subset.py --source indexfolder --root {out}")
    print("  python scripts/w4_baseline_accuracy.py      # ResNet-50 should be near 80% top-1")


if __name__ == "__main__":
    main()
