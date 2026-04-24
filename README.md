# Specification Dilemma Experiment

## Files

- `config.yaml` — LMStudio endpoint, model, generation and analysis parameters
- `prompts/sparse.json` — 30 sparse prompts
- `prompts/dense.json` — 30 dense prompts (matched to sparse by index)
- `smoke_test.py` — pre-flight: connectivity, seed-honoring, per-generation latency
- `generate.py` — runs completions against LMStudio
- `embed.py` — sentence embeddings
- `similarity.py` — pairwise cosine similarities
- `stats.py` — t-test, Mann-Whitney, bootstrap, Cohen's d
- `plot.py` — violin plot
- `run_all.py` — orchestrator (runs the five pipeline scripts in order)
- `pyproject.toml`, `uv.lock` — uv-managed environment
- `requirements.txt` — pip fallback
- `outputs/{sparse,dense}/NN.txt` — model completions (generated)
- `embeddings/{sparse,dense}.npy` — L2-normalized embedding matrices (generated)
- `results/pairwise.csv`, `results/stats.json`, `results/plot.png` — analysis artifacts (generated)

## Setup

1. Install LMStudio, load a strong instruction-tuned model, start the local server.
2. `uv sync`
3. Edit `config.yaml` for your LMStudio host, port, and model name.
4. `uv run python smoke_test.py` — verifies the endpoint and reports whether `seed` is honored.

## Run

```
uv run python run_all.py
```

Or step-by-step: `generate.py` → `embed.py` → `similarity.py` → `stats.py` → `plot.py`.
