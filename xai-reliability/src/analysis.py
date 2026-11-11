"""Weeks 9-10: ranking agreement between the two protocols.

RQ1 asks whether ROAD and IDSDS rank the five methods the same way; RQ2 asks
whether that ranking survives a change of architecture. Both reduce to Kendall's
tau between two orderings of the same five methods, which is why the scripts
write long-format rows: one row per (model, method, metric, seed, score).

Kendall's tau-b is used because it handles the ties that five-item rankings
produce. With only five methods per ranking the sampling noise is large, so the
tau values are reported over three seeds with a standard deviation, and read as
"do these protocols broadly agree" rather than as a precise estimate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def load_scores(results_dir: str | Path) -> pd.DataFrame:
    """Collect every scores_*.csv into one long table."""
    rows = []
    for path in sorted(Path(results_dir).glob("scores_*.csv")):
        rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError(f"no scores_*.csv under {results_dir}")
    return pd.concat(rows, ignore_index=True)


def method_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (model, metric, seed, method), ranked within each group.

    Rank 1 is the most faithful method. Both metrics are oriented so that
    higher is better, so ranking is always descending.
    """
    agg = df.groupby(["model", "metric", "seed", "method"], as_index=False)["score"].mean()
    agg["rank"] = agg.groupby(["model", "metric", "seed"])["score"].rank(ascending=False, method="average")
    return agg


def protocol_agreement(agg: pd.DataFrame, metric_a: str = "road", metric_b: str = "idsds") -> pd.DataFrame:
    """Kendall's tau between the two protocols, per model and seed (RQ1)."""
    out = []
    for (model, seed), block in agg.groupby(["model", "seed"]):
        a = block[block.metric == metric_a].set_index("method")["score"]
        b = block[block.metric == metric_b].set_index("method")["score"]
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 3:
            continue
        tau = kendalltau(a[common].values, b[common].values, variant="b")
        out.append(
            {
                "model": model,
                "seed": seed,
                "n_methods": len(common),
                "kendall_tau": tau.statistic,
                "p_value": tau.pvalue,
                "methods": ",".join(common),
            }
        )
    return pd.DataFrame(out)


def architecture_agreement(agg: pd.DataFrame, metric: str = "road") -> pd.DataFrame:
    """Kendall's tau between every pair of architectures (RQ2)."""
    out = []
    block = agg[agg.metric == metric]
    for seed, by_seed in block.groupby("seed"):
        pivot = by_seed.pivot_table(index="method", columns="model", values="score")
        models = list(pivot.columns)
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                pair = pivot[[m1, m2]].dropna()
                if len(pair) < 3:
                    continue
                tau = kendalltau(pair[m1].values, pair[m2].values, variant="b")
                out.append(
                    {
                        "metric": metric,
                        "seed": seed,
                        "model_a": m1,
                        "model_b": m2,
                        "n_methods": len(pair),
                        "kendall_tau": tau.statistic,
                        "p_value": tau.pvalue,
                    }
                )
    return pd.DataFrame(out)


def bcos_similarity(post_hoc: np.ndarray, builtin: np.ndarray) -> dict:
    """RQ3: how close a post-hoc map is to the B-cos built-in explanation.

    Two complementary views: Spearman correlation over all pixels (does the
    whole ranking agree) and top-10% overlap (do they agree about the evidence
    that matters).
    """
    a, b = post_hoc.reshape(len(post_hoc), -1), builtin.reshape(len(builtin), -1)
    rho, overlap = [], []
    for i in range(a.shape[0]):
        rho.append(float(spearmanr(a[i], b[i]).correlation))
        k = max(1, int(0.1 * a.shape[1]))
        top_a = set(np.argpartition(-a[i], k)[:k].tolist())
        top_b = set(np.argpartition(-b[i], k)[:k].tolist())
        overlap.append(len(top_a & top_b) / k)
    return {
        "spearman_mean": float(np.nanmean(rho)),
        "spearman_std": float(np.nanstd(rho)),
        "top10_overlap_mean": float(np.mean(overlap)),
        "top10_overlap_std": float(np.std(overlap)),
    }


def summary_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds, in the shape the report tables need."""
    out = agg.groupby(["model", "metric", "method"])["score"].agg(["mean", "std", "count"]).reset_index()
    out["rank"] = out.groupby(["model", "metric"])["mean"].rank(ascending=False, method="average")
    return out.sort_values(["metric", "model", "rank"])
