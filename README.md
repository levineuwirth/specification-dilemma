# Specification Dilemma Experiment

Small empirical probe for the claim that sparse-specification prompts
produce more homogeneous outputs across users than dense-specification
prompts.

See `experiment.pdf` for the full specification.

## Design note: matched pairs

`experiment.tex` describes a between-groups design with 30 independently-drawn
sparse prompts and 30 independently-drawn dense prompts. The prompts in this
repo follow a tighter variant: **matched pairs**. Each of 30 imagined users
has a fixed underlying intent (audience, thesis, tone, voice, opening move,
structural constraint). `prompts/dense.json[i]` expresses that user's full
intent; `prompts/sparse.json[i]` is what the same user would type when
underspecifying — topic only, in roughly their natural register. The sparse
prompts carry no audience, thesis, tone, or structural specification.

The statistical comparison is unchanged — cross-user pairwise similarity in
each condition — but the two conditions now sample the same population of
underlying intents. This tests the sharper claim: when users with divergent
intents underspecify, outputs converge (priors dominate); when they specify
fully, outputs diverge (intents dominate).

## Setup

1. Install LMStudio and download a strong instruction-tuned model
   (e.g. Qwen2.5-72B-Instruct or Llama-3.3-70B-Instruct).
2. Start the LMStudio local server (default: localhost:1234).
3. Create the environment and install dependencies with `uv`:

   ```
   uv sync
   ```

   (or `pip install -r requirements.txt` inside a venv if not using uv)

4. Edit `config.yaml` if your LMStudio model name or port differs from
   the defaults. If LMStudio is on a remote host, point `lmstudio.base_url`
   at that host (e.g. `http://<host>:1234/v1`).

5. Smoke-test the endpoint (checks connectivity, seed-honoring, and
   approximate per-generation latency):

   ```
   uv run python smoke_test.py
   ```

## Running

Freeze your prompts in `prompts/sparse.json` and `prompts/dense.json`
before generating anything.

Then run the full pipeline:

```
uv run python run_all.py
```

Or run steps individually:

```
uv run python generate.py     # LMStudio generations
uv run python embed.py        # sentence embeddings
uv run python similarity.py   # pairwise cosine similarities
uv run python stats.py        # t-test, Mann-Whitney, bootstrap, Cohen's d
uv run python plot.py         # violin plot
```

## Outputs

- `outputs/{sparse,dense}/NN.txt`    : raw model completions
- `embeddings/{sparse,dense}.npy`    : L2-normalized embedding matrices
- `results/pairwise.csv`             : all pairwise similarities
- `results/stats.json`               : test statistics and summary
- `results/plot.png`                 : similarity distribution plot

## Interpretation

A positive result: sparse-condition mean pairwise similarity is
meaningfully higher than dense-condition mean similarity, the
bootstrap 95% CI on the difference excludes 0, and Cohen's d is
large (>0.8).

A null or inverted result is also interesting and should be reported
honestly.
