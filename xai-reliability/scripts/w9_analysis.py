"""Weeks 9-10: ranking agreement, the headline tables and the figures.

    python scripts/w9_analysis.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _bootstrap  # noqa: F401
from src.analysis import (architecture_agreement, load_scores, method_rankings,
                          protocol_agreement, summary_table)
from src.config import load_config
from src.utils import ensure_dir, get_logger

LOG = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = cfg.resolve("paths.results")
    figures = ensure_dir(cfg.resolve("paths.figures"))

    scores = load_scores(results)
    agg = method_rankings(scores)
    agg.to_csv(results / "rankings.csv", index=False)

    table = summary_table(agg)
    table.to_csv(results / "summary_table.csv", index=False)
    print("\n== mean score by model, metric, method ==")
    print(table.to_string(index=False))

    rq1 = protocol_agreement(agg)
    rq1.to_csv(results / "rq1_protocol_agreement.csv", index=False)
    print("\n== RQ1: Kendall tau, ROAD vs IDSDS ==")
    print(rq1.to_string(index=False) if len(rq1) else "(need both metrics scored first)")

    rq2 = architecture_agreement(agg, metric="road")
    rq2.to_csv(results / "rq2_architecture_agreement.csv", index=False)
    print("\n== RQ2: Kendall tau between architectures (ROAD) ==")
    print(rq2.to_string(index=False) if len(rq2) else "(need two or more models scored first)")

    if len(rq1):
        means = rq1.groupby("model")["kendall_tau"].agg(["mean", "std"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(means.index, means["mean"], yerr=means["std"], capsize=4)
        ax.axhline(0, linewidth=0.8, color="black")
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Kendall's tau (ROAD vs IDSDS)")
        ax.set_title("Do the two faithfulness protocols agree?")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(figures / "rq1_protocol_agreement.png", dpi=200)
        LOG.info("figure -> %s", figures / "rq1_protocol_agreement.png")


if __name__ == "__main__":
    main()
