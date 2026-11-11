"""Week 6: ROAD - Remove And Debias (Rong et al., 2022).

Why not plain pixel deletion
----------------------------
If deleted pixels are set to a constant, the *shape* of the deleted region is
itself informative: a classifier can recover the class from the silhouette of
the mask, so the measured accuracy drop partly reflects leakage rather than the
quality of the attribution map. ROAD removes that channel by replacing each
deleted pixel with a noisy linear interpolation of its neighbours, so the
deleted region carries no shape information of its own.

The imputation solves, for the set of deleted pixels u,

    x_i = sum_j w_ij x_j + eps,   i in u

with the 3x3 neighbour weights of the reference implementation (1/6 for the
four edge neighbours, 1/12 for the four corners, summing to 1). Pixels outside
the image are dropped. Known neighbours move to the right-hand side, giving a
sparse linear system per colour channel.

Two orders are run, as in the paper:
    MoRF  most relevant first  - a good map gives a steep accuracy drop
    LeRF  least relevant first - a good map gives a flat curve
and the single faithfulness score reported here is

    ROAD = AUC(LeRF) - AUC(MoRF)        higher is better

which rewards a map that is right about both ends of the ranking.

Cost note: one sparse solve per image, per deletion fraction, per order. For a
first pass use --limit 200; the full 1,000-image run is an overnight CPU job
and parallelises cleanly over images.
"""
from __future__ import annotations

import atexit
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .models import ModelWrapper
from .utils import get_logger

LOG = get_logger()

# (dy, dx) -> weight, from the reference ROAD implementation.
NEIGHBOUR_WEIGHTS = {
    (-1, -1): 1 / 12, (-1, 0): 1 / 6, (-1, 1): 1 / 12,
    (0, -1): 1 / 6, (0, 1): 1 / 6,
    (1, -1): 1 / 12, (1, 0): 1 / 6, (1, 1): 1 / 12,
}


def noisy_linear_imputation(img: np.ndarray, keep: np.ndarray, noise_std: float = 0.01,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Impute the pixels where `keep` is False.

    img  : (C, H, W) float array
    keep : (H, W) boolean array, True for pixels that stay
    """
    rng = rng or np.random.default_rng(0)
    c, h, w = img.shape
    flat_keep = keep.reshape(-1)
    missing = np.flatnonzero(~flat_keep)
    if missing.size == 0:
        return img.copy()

    # Position of each missing pixel within the system we are about to build.
    slot = -np.ones(h * w, dtype=np.int64)
    slot[missing] = np.arange(missing.size)

    rows, cols, vals = [], [], []
    rhs = np.zeros((missing.size, c), dtype=np.float64)
    flat_img = img.reshape(c, -1)

    ys, xs = np.divmod(missing, w)
    for n, (y, x, idx) in enumerate(zip(ys, xs, missing)):
        rows.append(n); cols.append(n); vals.append(1.0)   # the x_i term
        for (dy, dx), weight in NEIGHBOUR_WEIGHTS.items():
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            nidx = ny * w + nx
            if flat_keep[nidx]:
                rhs[n] += weight * flat_img[:, nidx]        # known -> RHS
            else:
                rows.append(n); cols.append(slot[nidx]); vals.append(-weight)

    a = sp.csr_matrix((vals, (rows, cols)), shape=(missing.size, missing.size))
    rhs = rhs + rng.normal(0.0, noise_std, size=rhs.shape)

    out = img.reshape(c, -1).copy()
    for ch in range(c):
        try:
            solution = spla.spsolve(a, rhs[:, ch])
        except Exception:  # singular or ill-conditioned: fall back to iterative
            solution, _ = spla.cg(a, rhs[:, ch], rtol=1e-5, maxiter=500)
        out[ch, missing] = solution
    return np.clip(out.reshape(c, h, w), 0.0, 1.0)


def keep_mask_from_attribution(attr: np.ndarray, fraction: float, order: str) -> np.ndarray:
    """Which pixels survive when `fraction` of them is deleted.

    order = "morf": delete the highest-attribution pixels first.
    order = "lerf": delete the lowest-attribution pixels first.
    """
    h, w = attr.shape
    n_delete = int(round(fraction * h * w))
    if n_delete <= 0:
        return np.ones((h, w), dtype=bool)
    flat = attr.reshape(-1)
    # argsort is ascending, so LeRF deletes the head and MoRF the tail.
    ranked = np.argsort(flat, kind="stable")
    doomed = ranked[:n_delete] if order == "lerf" else ranked[-n_delete:]
    keep = np.ones(h * w, dtype=bool)
    keep[doomed] = False
    return keep.reshape(h, w)


@torch.no_grad()
def _predict(wrapper: ModelWrapper, batch: torch.Tensor) -> torch.Tensor:
    return wrapper.forward(batch).argmax(dim=1)


def _impute_one(task):
    """Worker for the process pool: one image at one deletion fraction."""
    img, attr, k, order, noise_std, seed = task
    keep = keep_mask_from_attribution(attr, k, order)
    return noisy_linear_imputation(img, keep, noise_std, np.random.default_rng(seed))


# One pool for the whole run. Spawning a fresh pool per batch re-imports torch
# in every worker each time, which is slow and, on Windows, a large memory spike.
_POOL: ProcessPoolExecutor | None = None
_POOL_JOBS = 0
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


def _get_pool(jobs: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_JOBS
    if _POOL is not None and _POOL_JOBS == jobs:
        return _POOL
    if _POOL is not None:
        _POOL.shutdown()
    # Parallelism comes from the processes, so each worker's BLAS must be
    # single-threaded. Otherwise every worker starts one BLAS thread per core
    # (jobs x cores threads in total) and OpenBLAS runs out of memory. Spawned
    # workers inherit these variables; the parent's BLAS is already loaded and
    # unaffected.
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    _POOL, _POOL_JOBS = ProcessPoolExecutor(max_workers=jobs), jobs
    return _POOL


@atexit.register
def _close_pool():
    if _POOL is not None:
        _POOL.shutdown()


def road_for_batch(wrapper: ModelWrapper, x: torch.Tensor, attributions: torch.Tensor,
                   targets: torch.Tensor, ks: list[float], orders: list[str],
                   noise_std: float = 0.01, seed: int = 0, jobs: int = 1) -> list[dict]:
    """Score one batch. Returns long-format rows, one per (order, k, image).

    The sparse solve dominates the runtime (roughly 0.5-0.7s per 224x224 image
    at a mid deletion fraction on one core) and is independent per image, so
    `jobs > 1` spreads it over a process pool. The forward pass stays on the
    main process, so the GPU is still called once per batch.
    """
    rows: list[dict] = []
    imgs = x.detach().cpu().numpy().astype(np.float64)
    attrs = attributions.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    # Shared across batches; closed at interpreter exit.
    pool = _get_pool(jobs) if jobs > 1 else None

    for order in orders:
        for k in ks:
            if pool is not None:
                tasks = [
                    (imgs[i], attrs[i], k, order, noise_std, seed * 100003 + i)
                    for i in range(imgs.shape[0])
                ]
                perturbed = np.stack(list(pool.map(_impute_one, tasks)))
            else:
                perturbed = np.empty_like(imgs)
                for i in range(imgs.shape[0]):
                    keep = keep_mask_from_attribution(attrs[i], k, order)
                    perturbed[i] = noisy_linear_imputation(imgs[i], keep, noise_std, rng)

            batch = torch.from_numpy(perturbed).float().to(wrapper.device)
            preds = _predict(wrapper, batch).cpu()
            for i in range(imgs.shape[0]):
                rows.append(
                    {
                        "order": order,
                        "k": k,
                        "correct": int(preds[i].item() == int(targets[i].item())),
                        "pred": int(preds[i].item()),
                    }
                )
    return rows


def summarise(df) -> dict:
    """Collapse per-image rows into the headline ROAD numbers.

    AUC is the mean accuracy across deletion fractions, which is the trapezoid
    rule on an evenly spaced grid up to a constant factor.
    """
    acc = df.groupby(["order", "k"])["correct"].mean().unstack("order")
    auc = acc.mean()
    morf, lerf = float(auc.get("morf", np.nan)), float(auc.get("lerf", np.nan))
    return {
        "auc_morf": morf,
        "auc_lerf": lerf,
        "road": lerf - morf,
        "accuracy_curve": acc.to_dict(),
    }
