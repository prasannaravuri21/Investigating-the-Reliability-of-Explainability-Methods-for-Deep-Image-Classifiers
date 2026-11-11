"""Week 7, step 1: fine-tune each model on images with random patches removed.

IDSDS only means anything if deletion is in-distribution, which is what this
fine-tune buys. The split used here must be disjoint from the evaluation
subset; build it with

    python scripts/w3_build_subset.py --source imagenetv2 --root <root> \
        --per-class 9 --exclude data/subset_manifest.csv --out data/finetune_manifest.csv

Top-1 accuracy on the clean evaluation subset is recorded before and after, so
the report can state exactly how much the fine-tune moved each model.

    python scripts/w7_finetune_patchdrop.py --models resnet50 bcos_resnet50
"""
import argparse

import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.config import load_config
from src.data import make_loader
from src.idsds import finetune_patchdrop
from src.models import load_model
from src.utils import append_rows, ensure_dir, get_device, get_logger, set_seed

LOG = get_logger()


@torch.no_grad()
def top1(wrapper, loader, device) -> float:
    hits = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        hits += (wrapper.forward(x).argmax(dim=1) == y).sum().item()
        total += y.numel()
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--train-manifest", default="data/finetune_manifest.csv")
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device(cfg.get_path("device", "auto"))
    grid = cfg.get_path("idsds.grid", 8)
    ft = cfg.get_path("idsds.finetune", {})
    ckpt_dir = ensure_dir(cfg.resolve("paths.checkpoints"))
    results = ensure_dir(cfg.resolve("paths.results"))

    for name in args.models or cfg["models"]:
        wrapper = load_model(name, device)
        eval_loader = make_loader(cfg.resolve("data.manifest"), wrapper.transform,
                                  ft.get("batch_size", 32), args.workers, args.eval_limit)
        before = top1(wrapper, eval_loader, device)

        train_loader = make_loader(args.train_manifest, wrapper.transform,
                                   ft.get("batch_size", 32), args.workers, shuffle=True)
        ckpt = str(ckpt_dir / f"{name}_patchdrop_seed{seed}.pt")
        finetune_patchdrop(
            wrapper, train_loader, grid=grid, n_drop=ft.get("n_drop", 1),
            epochs=ft.get("epochs", 1), lr=ft.get("lr", 1e-4),
            max_steps=ft.get("max_steps", 2000), checkpoint=ckpt, seed=seed,
            drop_prob=ft.get("drop_prob", 1.0),
        )
        after = top1(wrapper, eval_loader, device)

        append_rows(results / "idsds_finetune_log.csv", [{
            "model": name, "seed": seed, "grid": grid,
            "n_drop": ft.get("n_drop", 1), "drop_prob": ft.get("drop_prob", 1.0),
            "max_steps": ft.get("max_steps", 2000),
            "clean_top1_before": before, "clean_top1_after": after,
            "checkpoint": ckpt,
        }])
        LOG.info("%-16s clean top1 %.3f -> %.3f after patch-drop fine-tune", name, before, after)
        del wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
