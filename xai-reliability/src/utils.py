"""Small helpers shared by every script: seeding, device choice, logging, IO."""
from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Fix every RNG we touch. Called at the top of each script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs a little speed but makes a run repeatable,
    # which matters because the whole project is about measurement stability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# --------------------------------------------------------------------------- #
# GPU utilisation cap (duty cycle)
# --------------------------------------------------------------------------- #
# Laptop GPUs rarely allow a power or utilisation limit from software, so the
# cap is enforced here instead: after each short burst of GPU work the caller
# sleeps long enough that the GPU is busy only `_GPU_DUTY` of the time. Bursts
# are a fraction of a second, so monitors sampling once a second read a steady
# value near the cap rather than 100% spikes. Results are unaffected; only
# wall-clock time grows (by 1/duty).
_GPU_DUTY = 1.0


def set_gpu_duty_cycle(duty: float | None) -> None:
    global _GPU_DUTY
    _GPU_DUTY = 1.0 if duty is None else min(1.0, max(0.05, float(duty)))


def gpu_throttle(started: float, device: torch.device | str) -> None:
    """Pause after a burst of GPU work that began at `started` (perf_counter)."""
    if _GPU_DUTY >= 1.0 or torch.device(device).type != "cuda":
        return
    torch.cuda.synchronize(device)   # measure real GPU time, not launch time
    busy = time.perf_counter() - started
    time.sleep(busy * (1.0 / _GPU_DUTY - 1.0))


def get_logger(name: str = "xai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    log = logger or get_logger()
    start = time.time()
    yield
    log.info("%s took %.1fs", label, time.time() - start)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_rows(csv_path: str | Path, rows: list[dict]) -> Path:
    """Append rows to a CSV, writing the header only once.

    Every experiment writes long-format rows (one row per observation), which
    keeps the Week 9-10 analysis to a few groupby calls.
    """
    path = Path(csv_path)
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)
    return path
