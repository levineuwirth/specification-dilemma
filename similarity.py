"""Compute pairwise cosine similarities within each condition."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pairwise_cosine(embeddings: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (similarities, index_pairs) for all i<j pairs.

    Assumes embeddings are L2-normalized, so cosine = dot product.
    """
    n = embeddings.shape[0]
    pairs = list(combinations(range(n), 2))
    sims = np.array([
        float(embeddings[i] @ embeddings[j]) for i, j in pairs
    ])
    return sims, pairs


def main() -> None:
    cfg = load_config()
    emb_root = Path(cfg["paths"]["embeddings_dir"])
    results_root = Path(cfg["paths"]["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition in ("sparse", "dense"):
        emb_path = emb_root / f"{condition}.npy"
        if not emb_path.exists():
            print(f"Missing embeddings for {condition}; skipping.")
            continue
        embeddings = np.load(emb_path)
        sims, pairs = pairwise_cosine(embeddings)
        for (i, j), s in zip(pairs, sims):
            rows.append({
                "condition": condition,
                "i": i,
                "j": j,
                "cosine": s,
            })
        print(
            f"{condition}: n_outputs={embeddings.shape[0]}, "
            f"n_pairs={len(sims)}, mean={sims.mean():.4f}, "
            f"std={sims.std(ddof=1):.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(results_root / "pairwise.csv", index=False)
    print(f"Saved {results_root / 'pairwise.csv'}")


if __name__ == "__main__":
    main()
