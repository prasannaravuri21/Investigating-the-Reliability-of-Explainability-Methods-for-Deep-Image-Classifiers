"""Week 7, step 2: single-deletion scoring against the cached attribution maps.

Loads the patch-drop checkpoint written by w7_finetune_patchdrop.py, measures
the per-patch logit drop for every image, and correlates it with the per-patch
attribution. One row per image goes to results/idsds_per_image_*.csv, and the
mean goes to results/scores_idsds.csv in the same shape as the ROAD scores, so
the Week 9-10 analysis reads both with one function.

    python scripts/w5_run_attributions.py --models resnet50 --patchdrop --limit 200
    python scripts/w7_run_idsds.py --models resnet50 --limit 200

Following Hesse et al. (2024), the maps scored here explain the fine-tuned
model itself (<model>-patchdrop__<method>__seed<s>.npy, written by
w5_run_attributions.py --patchdrop), and the logit drops are measured for the
exact targets saved beside those maps. With --no-checkpoint the original model
and its original maps are scored instead, as the out-of-domain ablation.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.config import load_config
from src.data import make_loader
from src.idsds import (idsds_per_image, load_patchdrop_checkpoint, patch_attribution,
                       single_deletion_drops)
from src.models import load_model, resolve_target
from src.utils import append_rows, ensure_dir, get_device, get_logger, set_seed

LOG = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="score the un-fine-tuned model (out-of-domain; for the ablation only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device(cfg.get_path("device", "auto"))
    grid = cfg.get_path("idsds.grid", 8)
    fill = cfg.get_path("idsds.deletion_fill", "mean")
    batch_size = cfg.get_path("idsds.batch_size", 32)
    target_mode = cfg.get_path("attribution.target_mode", "predicted")
    manifest = cfg.resolve("data.manifest")
    results = ensure_dir(cfg.resolve("paths.results"))
    attr_dir = cfg.resolve("paths.attributions")

    for model_name in args.models or cfg["models"]:
        wrapper = load_model(model_name, device)
        ckpt = cfg.resolve("paths.checkpoints") / f"{model_name}_patchdrop_seed{seed}.pt"
        if args.no_checkpoint:
            LOG.warning("scoring %s WITHOUT the patch-drop fine-tune (out-of-domain)", model_name)
            tag = model_name
        elif ckpt.exists():
            load_patchdrop_checkpoint(wrapper, str(ckpt), expected_grid=grid)
            tag = f"{model_name}-patchdrop"
        else:
            LOG.error("no checkpoint %s - run w7_finetune_patchdrop.py first, "
                      "or pass --no-checkpoint for the ablation", ckpt)
            continue

        paths = sorted(p for p in attr_dir.glob(f"{tag}__*__seed{seed}.npy"))
        if args.methods:
            paths = [p for p in paths if p.name.split("__")[1] in args.methods]
        if not paths:
            LOG.error("no maps %s__*__seed%d.npy in %s - run w5_run_attributions.py%s first",
                      tag, seed, attr_dir, "" if args.no_checkpoint else " --patchdrop")
            continue

        # Targets saved with the maps: the drops must be measured for the class
        # each map explains. All methods share them (same model, same images).
        def saved_targets(path):
            t = path.with_name(path.name.replace(".npy", "__targets.npy"))
            return np.load(t) if t.exists() else None
        ref_targets = saved_targets(paths[0])
        if ref_targets is None:
            LOG.warning("no saved targets for %s; resolving with mode=%s", paths[0].name, target_mode)

        # The logit drops do not depend on the attribution method, so compute
        # them once per model and reuse them for all methods.
        loader = make_loader(manifest, wrapper.transform, batch_size, args.workers, args.limit)
        drops, ids, cursor = [], [], 0
        for x, y, image_id in tqdm(loader, desc=f"deletions {tag}"):
            x, y = x.to(device), y.to(device)
            n = x.shape[0]
            if ref_targets is not None:
                if cursor + n > len(ref_targets):
                    raise RuntimeError(f"{paths[0].name} covers {len(ref_targets)} images but more "
                                       "were loaded; pass the same --limit used for the maps")
                target = torch.from_numpy(ref_targets[cursor: cursor + n]).long().to(device)
            else:
                target = resolve_target(wrapper, x, y, target_mode)
            cursor += n
            drops.append(single_deletion_drops(wrapper, x, target, grid, fill, batch_size))
            ids.append(image_id.numpy())
        drops = torch.cat(drops)
        ids = np.concatenate(ids)

        for path in paths:
            method = path.name.split("__")[1]
            t = saved_targets(path)
            if ref_targets is not None and t is not None and not np.array_equal(t[: len(ids)], ref_targets[: len(ids)]):
                LOG.warning("%s explains different targets than %s; its drops do not match",
                            path.name, paths[0].name)
            maps = torch.from_numpy(np.load(path).astype(np.float32))[: len(ids)]
            patch_attr = patch_attribution(maps, grid)
            scores = idsds_per_image(drops[: len(maps)], patch_attr)

            per_image = pd.DataFrame({"image_id": ids[: len(scores)], "idsds": scores})
            per_image.to_csv(results / f"idsds_per_image_{tag}_{method}_seed{seed}.csv", index=False)

            mean = float(np.nanmean(scores))
            append_rows(results / "scores_idsds.csv", [{
                "model": model_name, "method": method, "metric": "idsds", "seed": seed,
                "target_mode": target_mode,
                "score": mean, "std": float(np.nanstd(scores)),
                "n_images": int(np.sum(~np.isnan(scores))), "grid": grid,
            }])
            LOG.info("%-16s %-22s IDSDS %.4f", model_name, method, mean)

        del wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
