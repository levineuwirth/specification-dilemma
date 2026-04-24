"""Compute sentence embeddings for each generation in each condition."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_outputs(outputs_dir: Path) -> list[str]:
    """Load all .txt outputs from a condition directory, sorted by filename."""
    files = sorted(outputs_dir.glob("*.txt"))
    return [f.read_text(encoding="utf-8") for f in files]


def embed_condition(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    """Return (N, D) embedding matrix. L2-normalized for cosine similarity."""
    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def main() -> None:
    cfg = load_config()
    model = SentenceTransformer(cfg["embedding"]["model"])

    outputs_root = Path(cfg["paths"]["outputs_dir"])
    emb_root = Path(cfg["paths"]["embeddings_dir"])
    emb_root.mkdir(parents=True, exist_ok=True)

    for condition in ("sparse", "dense"):
        texts = load_outputs(outputs_root / condition)
        if not texts:
            print(f"No outputs found for {condition}; skipping.")
            continue
        print(f"Embedding {len(texts)} {condition} outputs...")
        embeddings = embed_condition(model, texts)
        np.save(emb_root / f"{condition}.npy", embeddings)
        print(f"Saved {condition}.npy with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
