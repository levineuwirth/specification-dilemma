"""Statistical tests and summary metrics for the similarity comparison.

Reports:
  - Per-condition descriptive statistics
  - Naive two-sample t-test on pairwise similarities
  - Mann-Whitney U (nonparametric check)
  - Cohen's d effect size
  - Output-level bootstrap 95% CI for the difference in mean similarity
    (corrects for pairwise dependence)
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d for two independent samples."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd


def mean_pairwise(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine for an L2-normalized embedding matrix."""
    n = embeddings.shape[0]
    sims = [
        float(embeddings[i] @ embeddings[j])
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(sims))


def bootstrap_diff(
    sparse_emb: np.ndarray,
    dense_emb: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Output-level bootstrap of (mean_sparse - mean_dense).

    Resamples outputs (not pairs) with replacement, recomputes mean
    pairwise similarity in each condition, returns 95% CI.
    """
    n_s, n_d = sparse_emb.shape[0], dense_emb.shape[0]
    diffs = np.empty(n_iter, dtype=float)
    for k in range(n_iter):
        idx_s = rng.integers(0, n_s, size=n_s)
        idx_d = rng.integers(0, n_d, size=n_d)
        ms = mean_pairwise(sparse_emb[idx_s])
        md = mean_pairwise(dense_emb[idx_d])
        diffs[k] = ms - md
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), diffs


def main() -> None:
    cfg = load_config()
    emb_root = Path(cfg["paths"]["embeddings_dir"])
    results_root = Path(cfg["paths"]["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)

    sparse_emb = np.load(emb_root / "sparse.npy")
    dense_emb = np.load(emb_root / "dense.npy")

    df = pd.read_csv(results_root / "pairwise.csv")
    sparse_sims = df.loc[df["condition"] == "sparse", "cosine"].to_numpy()
    dense_sims = df.loc[df["condition"] == "dense", "cosine"].to_numpy()

    # Descriptive
    desc = {
        "sparse": {
            "n_outputs": int(sparse_emb.shape[0]),
            "n_pairs": int(len(sparse_sims)),
            "mean": float(sparse_sims.mean()),
            "std": float(sparse_sims.std(ddof=1)),
            "median": float(np.median(sparse_sims)),
        },
        "dense": {
            "n_outputs": int(dense_emb.shape[0]),
            "n_pairs": int(len(dense_sims)),
            "mean": float(dense_sims.mean()),
            "std": float(dense_sims.std(ddof=1)),
            "median": float(np.median(dense_sims)),
        },
    }

    # Naive t-test (note: pairwise dependence means this is optimistic)
    t_stat, t_p = stats.ttest_ind(sparse_sims, dense_sims, equal_var=False)

    # Nonparametric check
    u_stat, u_p = stats.mannwhitneyu(
        sparse_sims, dense_sims, alternative="two-sided"
    )

    # Effect size
    d = cohens_d(sparse_sims, dense_sims)

    # Output-level bootstrap (the honest test)
    rng = np.random.default_rng(cfg["analysis"]["random_seed"])
    lo, hi, _ = bootstrap_diff(
        sparse_emb,
        dense_emb,
        n_iter=cfg["analysis"]["bootstrap_iterations"],
        rng=rng,
    )

    summary = {
        "descriptive": desc,
        "naive_welch_t_test": {"t": float(t_stat), "p": float(t_p)},
        "mann_whitney_u": {"u": float(u_stat), "p": float(u_p)},
        "cohens_d": float(d),
        "bootstrap_diff_in_means": {
            "point_estimate": float(sparse_sims.mean() - dense_sims.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "n_iter": int(cfg["analysis"]["bootstrap_iterations"]),
        },
    }

    with open(results_root / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty print
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
