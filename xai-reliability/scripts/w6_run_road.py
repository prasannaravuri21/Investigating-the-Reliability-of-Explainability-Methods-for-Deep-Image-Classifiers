"""Week 6: score every cached attribution map with ROAD.

Writes two files:
  results/road_curves_<model>_<method>_seed<s>.csv   per-image, per-k rows
  results/scores_road.csv                           one headline row per cell

Start with --limit 200 to get a first ROAD curve inside an afternoon (the IPR
targets exactly that for early Week 5-6); the full 1,000-image run is an
overnight CPU job.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.config import load_config
from src.data import make_loader
from src.models import load_model, resolve_target
from src.road import road_for_batch, summarise
from src.utils import append_rows, ensure_dir, get_device, get_logger, set_seed

LOG = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers")
    ap.add_argument("--jobs", type=int, default=1,
                    help="processes for the sparse imputation solve; set to your core count")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device(cfg.get_path("device", "auto"))
    manifest = cfg.resolve("data.manifest")
    results = ensure_dir(cfg.resolve("paths.results"))
    attr_dir = cfg.resolve("paths.attributions")
    ks = cfg.get_path("road.ks")
    orders = cfg.get_path("road.orders", ["morf", "lerf"])
    noise_std = cfg.get_path("road.noise_std", 0.01)
    batch_size = cfg.get_path("road.batch_size", 16)
    target_mode = cfg.get_path("attribution.target_mode", "predicted")

    for model_name in args.models or cfg["models"]:
        wrapper = load_model(model_name, device)
        available = sorted(p for p in attr_dir.glob(f"{model_name}__*__seed{seed}.npy") if "ids" not in p.name)
        for path in available:
            method = path.name.split("__")[1]
            if args.methods and method not in args.methods:
                continue

            maps = np.load(path).astype(np.float32)
            # Reuse the exact targets the maps were produced for; without this,
            # "accuracy after deletion" would be measured against a different
            # class than the one explained.
            tgt_path = path.with_name(path.name.replace(".npy", "__targets.npy"))
            saved_targets = np.load(tgt_path) if tgt_path.exists() else None
            if saved_targets is None:
                LOG.warning("no saved targets for %s; recomputing with mode=%s",
                            path.name, target_mode)
            loader = make_loader(manifest, wrapper.transform, batch_size, args.workers, args.limit)
            rows, cursor = [], 0
            for x, y, image_id in tqdm(loader, desc=f"ROAD {model_name}/{method}"):
                n = x.shape[0]
                attr = torch.from_numpy(maps[cursor: cursor + n])
                if saved_targets is not None:
                    target = torch.from_numpy(saved_targets[cursor: cursor + n]).long()
                else:
                    target = resolve_target(wrapper, x.to(device), y.to(device), target_mode).cpu()
                cursor += n
                batch_rows = road_for_batch(wrapper, x.to(device), attr, target, ks, orders,
                                           noise_std, seed, jobs=args.jobs)
                # road_for_batch returns rows ordered (order, k, image); attach ids
                per_cell = n
                for j, row in enumerate(batch_rows):
                    row["image_id"] = int(image_id[j % per_cell].item())
                    row["model"] = model_name
                    row["method"] = method
                    row["seed"] = seed
                rows.extend(batch_rows)

            df = pd.DataFrame(rows)
            curve_path = results / f"road_curves_{model_name}_{method}_seed{seed}.csv"
            df.to_csv(curve_path, index=False)

            stats = summarise(df)
            append_rows(
                results / "scores_road.csv",
                [{
                    "model": model_name, "method": method, "metric": "road", "seed": seed,
                    "target_mode": target_mode, "score": stats["road"], "auc_morf": stats["auc_morf"], "auc_lerf": stats["auc_lerf"],
                    "n_images": int(df.image_id.nunique()),
                }],
            )
            LOG.info("%-16s %-22s ROAD %.4f (MoRF %.3f / LeRF %.3f)",
                     model_name, method, stats["road"], stats["auc_morf"], stats["auc_lerf"])

        del wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
