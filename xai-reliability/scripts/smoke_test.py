"""Run this first, on the machine that will do the real work.

It builds the architectures with RANDOM weights (no 500 MB of checkpoints to
download), generates a handful of synthetic images, and pushes them through
every stage: attribution -> ROAD -> patch-drop fine-tune -> IDSDS -> analysis.
The numbers are meaningless; the point is that each stage executes and the
shapes line up before any GPU time is spent.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --models resnet50 vit_b_16

Expected: every line prints "ok", except LRP on the ViT, which prints
"not applicable" by design.
"""
import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("XAI_PRETRAINED", "0")   # must be set before src.models imports

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401


def make_fake_dataset(root: Path, n_classes: int = 4, per_class: int = 2) -> Path:
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        folder = root / str(c)
        folder.mkdir(parents=True, exist_ok=True)
        for j in range(per_class):
            arr = (rng.random((300, 300, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(folder / f"img{j}.jpeg")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["resnet50"])
    ap.add_argument("--methods", nargs="*",
                    default=["gradcam", "integrated_gradients", "lrp", "rise", "gradient_shap"])
    ap.add_argument("--n-images", type=int, default=4)
    args = ap.parse_args()

    import torch

    from src.attributions import BUILTIN, compute_attribution
    from src.config import load_config
    from src.data import build_manifest, make_loader
    from src.idsds import (drop_random_patches, fill_value, idsds_per_image,
                           patch_attribution, single_deletion_drops)
    from src.models import load_model
    from src.road import road_for_batch, summarise
    from src.utils import get_device, set_seed

    import pandas as pd

    cfg = load_config()
    set_seed(0)
    device = get_device("cpu")   # the smoke test stays on CPU on purpose
    print(f"device: {device}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = make_fake_dataset(tmp / "fake")
        manifest = tmp / "manifest.csv"
        build_manifest("imagenetv2", root, manifest, per_class=1, seed=0)
        print(f"ok   manifest built ({len(pd.read_csv(manifest))} images)")

        for model_name in args.models:
            print(f"\n--- {model_name} (random weights) ---")
            wrapper = load_model(model_name, device)
            loader = make_loader(manifest, wrapper.transform, batch_size=args.n_images, num_workers=0)
            x, y, _ = next(iter(loader))
            x, y = x.to(device), y.to(device)
            print(f"ok   forward pass {tuple(x.shape)} -> {tuple(wrapper.forward(x).shape)}")

            methods = list(args.methods) + ([BUILTIN] if wrapper.is_bcos else [])
            maps = {}
            for method in methods:
                try:
                    attr = compute_attribution(method, wrapper, x, y, cfg)
                    assert attr.shape == (x.shape[0], x.shape[2], x.shape[3]), attr.shape
                    assert torch.isfinite(attr).all(), f"{method} produced non-finite values"
                    maps[method] = attr
                    print(f"ok   {method:22s} map {tuple(attr.shape)} "
                          f"range [{attr.min():.3g}, {attr.max():.3g}]")
                except NotImplementedError as exc:
                    print(f"--   {method:22s} not applicable: {str(exc).splitlines()[0]}")

            method = next(iter(maps))
            rows = road_for_batch(wrapper, x, maps[method], y.cpu(), ks=[0.3, 0.7],
                                  orders=["morf", "lerf"], noise_std=0.01, seed=0)
            for i, row in enumerate(rows):
                row["image_id"] = i % x.shape[0]
            stats = summarise(pd.DataFrame(rows))
            print(f"ok   ROAD on {method}: {stats['road']:+.3f} "
                  f"(MoRF {stats['auc_morf']:.2f} / LeRF {stats['auc_lerf']:.2f})")

            grid = 4   # keep the smoke test cheap
            dropped = drop_random_patches(x, grid, 1, fill_value("mean", device))
            assert dropped.shape == x.shape
            print(f"ok   patch-drop augmentation, grid {grid}x{grid}")

            drops = single_deletion_drops(wrapper, x, y, grid=grid, batch_size=8)
            patch_attr = patch_attribution(maps[method], grid=grid)
            assert drops.shape == patch_attr.shape == (x.shape[0], grid * grid)
            scores = idsds_per_image(drops, patch_attr)
            print(f"ok   IDSDS on {method}: {np.nanmean(scores):+.3f} "
                  f"(per-image {['%.2f' % s for s in scores]})")

            del wrapper

        # FunnyBirds part scoring: a three-colour part map with all the
        # attribution mass inside one part must attribute it to that part.
        from src.funnybirds import (important_parts_at_thresholds,
                                    part_importance_from_map)

        part_map = torch.zeros(3, 32, 32)
        part_map[0, :10, :] = 1.0        # red    -> beak
        part_map[1, 10:20, :] = 1.0      # green  -> wing
        part_map[2, 20:, :] = 1.0        # blue   -> tail
        colors_to_part = {(1, 0, 0): "beak", (0, 1, 0): "wing", (0, 0, 1): "tail"}

        attr = torch.zeros(32, 32)
        attr[12:18, 4:28] = 1.0          # all the evidence sits on the wing
        importance = part_importance_from_map(attr, part_map, colors_to_part)
        assert importance["wing"] > 0 and importance["beak"] == 0, importance
        # The wing holds 100% of the evidence, so it clears every threshold
        # below 1.0; nothing can exceed a share of 1.0.
        top = important_parts_at_thresholds(importance, [0.0, 0.5, 1.0])
        assert top[0] == ["wing"] and top[1] == ["wing"] and top[2] == [], top
        print(f"\nok   FunnyBirds part scoring: {importance} -> important={top[0]}")

        from src.analysis import method_rankings, protocol_agreement
        fake = pd.DataFrame([
            {"model": "m", "method": f"meth{i}", "metric": metric, "seed": 0, "score": float(i)}
            for metric in ("road", "idsds") for i in range(5)
        ])
        agg = method_rankings(fake)
        tau = protocol_agreement(agg)
        print(f"\nok   analysis: Kendall tau on a perfectly agreeing pair = "
              f"{tau.kendall_tau.iloc[0]:.2f} (expected 1.00)")

    print("\nSMOKE TEST PASSED - the pipeline runs end to end. "
          "Now unset XAI_PRETRAINED and run the real Week 3-4 scripts.")


if __name__ == "__main__":
    main()
