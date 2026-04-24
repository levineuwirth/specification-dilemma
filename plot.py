"""Violin + strip plot of pairwise similarity distributions per condition."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    results_root = Path(cfg["paths"]["results_dir"])
    df = pd.read_csv(results_root / "pairwise.csv")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(
        data=df,
        x="condition",
        y="cosine",
        order=["sparse", "dense"],
        inner="quartile",
        cut=0,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="condition",
        y="cosine",
        order=["sparse", "dense"],
        color="black",
        alpha=0.25,
        size=2,
        ax=ax,
    )

    ax.set_xlabel("Specification condition")
    ax.set_ylabel("Pairwise cosine similarity")
    ax.set_title("Output similarity by specification density")
    fig.tight_layout()
    fig.savefig(results_root / "plot.png", dpi=200)
    print(f"Saved {results_root / 'plot.png'}")


if __name__ == "__main__":
    main()
