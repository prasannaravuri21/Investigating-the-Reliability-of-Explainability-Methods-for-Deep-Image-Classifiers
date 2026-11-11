"""Week 4: load the four models and record top-1 accuracy on the subset.

This is the sanity gate for everything after it. If a model's top-1 on the
subset is far from its published figure, the preprocessing or the class-index
mapping is wrong, and no faithfulness score computed on top of it would mean
anything.

    python scripts/w4_baseline_accuracy.py
    python scripts/w4_baseline_accuracy.py --models resnet50 --limit 100
"""
import argparse

import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.config import load_config
from src.data import make_loader
from src.models import load_model
from src.utils import append_rows, ensure_dir, get_device, get_logger, set_seed

LOG = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get_path("device", "auto"))
    manifest = cfg.resolve("data.manifest")
    out = ensure_dir(cfg.resolve("paths.results")) / "baseline_accuracy.csv"

    for name in args.models or cfg["models"]:
        wrapper = load_model(name, device)
        loader = make_loader(manifest, wrapper.transform, args.batch_size, args.workers, args.limit)
        top1 = top5 = total = 0
        with torch.no_grad():
            for x, y, _ in tqdm(loader, desc=name):
                x, y = x.to(device), y.to(device)
                logits = wrapper.forward(x)
                _, pred5 = logits.topk(5, dim=1)
                top1 += (pred5[:, 0] == y).sum().item()
                top5 += (pred5 == y.view(-1, 1)).any(dim=1).sum().item()
                total += y.numel()
        row = {
            "model": name,
            "n_images": total,
            "top1": top1 / total,
            "top5": top5 / total,
            "notes": wrapper.notes,
        }
        append_rows(out, [row])
        LOG.info("%-16s top1 %.3f  top5 %.3f  (n=%d)", name, row["top1"], row["top5"], total)
        del wrapper
        torch.cuda.empty_cache()

    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
