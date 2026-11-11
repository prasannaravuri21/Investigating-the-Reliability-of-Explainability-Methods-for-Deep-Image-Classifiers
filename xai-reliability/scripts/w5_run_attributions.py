"""Week 5: compute and cache the attribution maps.

Twenty method-model combinations, plus the B-cos built-in explanation. Maps are
cached as float16 .npy (about 100 MB per combination at 224x224 for 1,000
images) because both ROAD and IDSDS read the same maps, and the qualitative
figures in Chapter 4 read them again.

    python scripts/w5_run_attributions.py --models resnet50 --methods gradcam --limit 50
    python scripts/w5_run_attributions.py            # everything in the config

    # IDSDS maps: explain the patch-drop fine-tuned model, as Hesse et al. do
    python scripts/w5_run_attributions.py --models resnet50 --patchdrop --limit 50

ROAD scores the original models, so it reads <model>__<method>__seed<s>.npy.
IDSDS scores the fine-tuned models, so --patchdrop writes
<model>-patchdrop__<method>__seed<s>.npy, which w7_run_idsds.py reads. Each
protocol therefore scores maps of the model it actually measures.
"""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
from src.attributions import BUILTIN, compute_attribution
from src.config import load_config
from src.data import make_loader
from src.idsds import load_patchdrop_checkpoint
from src.models import load_model, resolve_target
from src.utils import append_rows, ensure_dir, get_device, get_logger, gpu_throttle, set_seed

LOG = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--patchdrop", action="store_true",
                    help="explain the patch-drop fine-tuned model (for IDSDS) instead of the original")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    device = get_device(cfg.get_path("device", "auto"))
    manifest = cfg.resolve("data.manifest")
    attr_dir = ensure_dir(cfg.resolve("paths.attributions"))
    log_csv = ensure_dir(cfg.resolve("paths.results")) / "attribution_runs.csv"
    batch_size = cfg.get_path("attribution.batch_size", 8)
    target_mode = cfg.get_path("attribution.target_mode", "predicted")

    for model_name in args.models or cfg["models"]:
        wrapper = load_model(model_name, device)
        tag = model_name
        if args.patchdrop:
            ckpt = cfg.resolve("paths.checkpoints") / f"{model_name}_patchdrop_seed{seed}.pt"
            if not ckpt.exists():
                LOG.error("no checkpoint %s - run w7_finetune_patchdrop.py first", ckpt)
                continue
            load_patchdrop_checkpoint(wrapper, str(ckpt), expected_grid=cfg.get_path("idsds.grid"))
            tag = f"{model_name}-patchdrop"
        methods = list(args.methods or cfg["methods"])
        if wrapper.is_bcos and BUILTIN not in methods:
            methods.append(BUILTIN)   # the reference explanation for RQ3

        loader = make_loader(manifest, wrapper.transform, batch_size, args.workers, args.limit)

        for method in methods:
            out_path = attr_dir / f"{tag}__{method}__seed{seed}.npy"
            if out_path.exists() and not args.overwrite:
                LOG.info("skip %s (exists)", out_path.name)
                continue

            maps, ids, targets, started, failed = [], [], [], time.time(), None
            try:
                for x, y, image_id in tqdm(loader, desc=f"{tag}/{method}"):
                    burst = time.perf_counter()
                    x, y = x.to(device), y.to(device)
                    # The same target must be used when the map is scored, so it
                    # is saved next to the map rather than recomputed later.
                    target = resolve_target(wrapper, x, y, target_mode)
                    attr = compute_attribution(method, wrapper, x, target, cfg)
                    maps.append(attr.detach().cpu().numpy().astype(np.float16))
                    ids.append(image_id.numpy())
                    targets.append(target.cpu().numpy())
                    if method != "rise":   # RISE throttles itself, per mask batch
                        gpu_throttle(burst, device)
            except NotImplementedError as exc:
                # LRP on the ViT lands here by design; record it rather than
                # quietly producing a number that is not LRP.
                failed = str(exc)
                LOG.warning("%s/%s not applicable: %s", tag, method, exc)

            elapsed = time.time() - started
            if failed is None:
                np.save(out_path, np.concatenate(maps, axis=0))
                np.save(attr_dir / f"{tag}__{method}__seed{seed}__ids.npy", np.concatenate(ids))
                np.save(attr_dir / f"{tag}__{method}__seed{seed}__targets.npy",
                        np.concatenate(targets))
                LOG.info("saved %s  (%.1fs)", out_path.name, elapsed)

            append_rows(
                log_csv,
                [{
                    "model": tag, "method": method, "seed": seed,
                    "target_mode": target_mode,
                    "n_images": int(sum(len(m) for m in maps)),
                    "seconds": round(elapsed, 1),
                    "status": "ok" if failed is None else "not_applicable",
                    "message": failed or "",
                }],
            )

        del wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
