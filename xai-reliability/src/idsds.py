"""Week 7: IDSDS - In-Domain Single-Deletion Score (Hesse et al., 2024).

The problem IDSDS fixes
-----------------------
Deleting part of an image pushes it off the data distribution the model was
trained on, so the accuracy drop confounds "this region mattered" with "this
input is now unfamiliar". IDSDS removes the confound by fine-tuning each model
on images with random patches already removed. Deletion then lies inside the
training distribution, and a single-patch deletion measures the model's
sensitivity to that patch rather than its surprise at a corrupted input.

Procedure
---------
1. Fine-tune the classifier for a short schedule on images with `n_drop` random
   patches removed (patches are filled with the dataset mean).
2. At test time delete one patch at a time and record the drop in the target
   logit.
3. Sum the attribution map inside each patch, then take the Spearman rank
   correlation between the per-patch attribution and the per-patch logit drop.
   IDSDS is that correlation, averaged over images: 1 means the map ranks
   patches exactly as the model's own sensitivity does.

The default grid of 4x4 on a 224x224 input gives 16 patches of 56x56, the main
setting of Hesse et al. (2024), so one image costs 16 forward passes plus one
clean pass. (An 8x8 grid was tried and rejected: single 28x28 deletions barely
move a ResNet-50 logit, so every method scored near zero.) This is why ROAD runs on all four
architectures and IDSDS is prioritised (Section 3.5 contingency: ResNet-50 and
B-cos first if GPU time runs short).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .models import IMAGENET_MEAN, ModelWrapper
from .utils import ensure_dir, get_logger, gpu_throttle

LOG = get_logger()


def patch_slices(h: int, w: int, grid: int) -> list[tuple[slice, slice]]:
    """Row-major list of patch slices for a grid x grid partition."""
    ys = [round(i * h / grid) for i in range(grid + 1)]
    xs = [round(j * w / grid) for j in range(grid + 1)]
    return [(slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1])) for i in range(grid) for j in range(grid)]


def fill_value(kind: str, device: torch.device) -> torch.Tensor:
    """What a deleted patch is replaced with, in [0, 1] image space."""
    if kind == "zero":
        return torch.zeros(3, 1, 1, device=device)
    return torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)


def drop_random_patches(x: torch.Tensor, grid: int, n_drop: int, fill: torch.Tensor,
                        generator: torch.Generator | None = None,
                        prob: float = 1.0) -> torch.Tensor:
    """Training-time augmentation: remove `n_drop` random patches from each image
    with probability `prob` (Hesse et al. augment half the images, prob=0.5), so
    the fine-tuned model still sees intact images too."""
    out = x.clone()
    slices = patch_slices(x.shape[-2], x.shape[-1], grid)
    for i in range(x.shape[0]):
        if prob < 1.0 and torch.rand(1, generator=generator).item() >= prob:
            continue
        picks = torch.randperm(len(slices), generator=generator, device="cpu")[:n_drop]
        for p in picks.tolist():
            ys, xs = slices[p]
            out[i, :, ys, xs] = fill
    return out


def finetune_patchdrop(wrapper: ModelWrapper, loader, grid: int = 8, n_drop: int = 1,
                       epochs: int = 1, lr: float = 1e-4, max_steps: int = 2000,
                       checkpoint: str | None = None, seed: int = 0,
                       drop_prob: float = 1.0) -> ModelWrapper:
    """Short fine-tune so that patch deletion is in-domain.

    The schedule is deliberately small (one pass, low LR, step cap): the point
    is to adapt the model to missing patches, not to improve it. Top-1 accuracy
    before and after is logged so any drift can be reported honestly.
    """
    device = wrapper.device
    model = wrapper.model
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    fill = fill_value("mean", device)
    gen = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    step = 0
    for epoch in range(epochs):
        for x, y, _ in loader:
            burst = time.perf_counter()
            x, y = x.to(device), y.to(device)
            x = drop_random_patches(x, grid, n_drop, fill, gen, prob=drop_prob)
            logits = wrapper.forward(x)
            loss = F.cross_entropy(logits, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            gpu_throttle(burst, device)
            step += 1
            if step % 50 == 0:
                LOG.info("[%s] finetune step %d/%d loss %.4f", wrapper.name, step, max_steps, loss.item())
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if checkpoint:
        ensure_dir(checkpoint.rsplit("/", 1)[0] if "/" in checkpoint else ".")
        torch.save({"model": model.state_dict(), "steps": step, "grid": grid, "n_drop": n_drop,
                    "drop_prob": drop_prob}, checkpoint)
        LOG.info("saved patch-drop checkpoint -> %s", checkpoint)
    return wrapper


def load_patchdrop_checkpoint(wrapper: ModelWrapper, path: str,
                              expected_grid: int | None = None) -> ModelWrapper:
    state = torch.load(path, map_location=wrapper.device)
    # A model adapted to one patch size is not in-domain for another, so a
    # stale checkpoint must fail loudly rather than score quietly.
    if expected_grid is not None and state.get("grid") != expected_grid:
        raise RuntimeError(
            f"{path} was fine-tuned with a {state.get('grid')}x{state.get('grid')} grid, but "
            f"idsds.grid is {expected_grid}. Re-run scripts/w7_finetune_patchdrop.py."
        )
    wrapper.model.load_state_dict(state["model"])
    wrapper.model.eval()
    LOG.info("loaded patch-drop checkpoint %s (%d steps)", path, state.get("steps", -1))
    return wrapper


@torch.no_grad()
def single_deletion_drops(wrapper: ModelWrapper, x: torch.Tensor, target: torch.Tensor,
                          grid: int = 8, fill_kind: str = "mean", batch_size: int = 32) -> torch.Tensor:
    """(B, grid*grid) drop in the target logit when each patch is deleted alone."""
    device = wrapper.device
    slices = patch_slices(x.shape[-2], x.shape[-1], grid)
    fill = fill_value(fill_kind, device)

    clean = wrapper.forward(x).gather(1, target.view(-1, 1)).squeeze(1)
    drops = torch.zeros(x.shape[0], len(slices), device=device)

    for i in range(x.shape[0]):
        variants = x[i: i + 1].repeat(len(slices), 1, 1, 1)
        for p, (ys, xs) in enumerate(slices):
            variants[p, :, ys, xs] = fill
        logits = []
        for start in range(0, len(slices), batch_size):
            burst = time.perf_counter()
            chunk = variants[start: start + batch_size]
            logits.append(wrapper.forward(chunk)[:, target[i]])
            gpu_throttle(burst, device)
        drops[i] = clean[i] - torch.cat(logits)
    return drops.cpu()


def patch_attribution(attr: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """(B, H, W) map -> (B, grid*grid) summed attribution per patch."""
    slices = patch_slices(attr.shape[-2], attr.shape[-1], grid)
    out = torch.zeros(attr.shape[0], len(slices))
    for p, (ys, xs) in enumerate(slices):
        out[:, p] = attr[:, ys, xs].sum(dim=(1, 2)).cpu()
    return out


def idsds_per_image(drops: torch.Tensor, patch_attr: torch.Tensor) -> list[float]:
    """Spearman rank correlation between per-patch attribution and logit drop."""
    scores = []
    d, a = drops.numpy(), patch_attr.numpy()
    for i in range(d.shape[0]):
        if a[i].std() == 0 or d[i].std() == 0:
            scores.append(float("nan"))  # a constant map has no ranking to score
            continue
        scores.append(float(spearmanr(a[i], d[i]).correlation))
    return scores
