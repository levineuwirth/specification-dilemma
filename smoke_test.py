"""Smoke-test the LMStudio endpoint before running the full pipeline.

Verifies:
  - The server is reachable at the configured base_url
  - A minimal chat completion returns content
  - The `seed` parameter is honored (two identical requests produce
    identical outputs)

This is a pre-flight check, not part of the pipeline. Run once after
pointing config.yaml at your LMStudio host, before `python run_all.py`.

Usage:
  python smoke_test.py
"""
from __future__ import annotations

import time

from generate import load_config, make_client


SMOKE_PROMPT = "In one sentence, describe a glass of water on a table."
SMOKE_SEED = 0
SMOKE_MAX_TOKENS = 80


def main() -> None:
    cfg = load_config()
    gen_cfg = cfg["generation"]
    model = cfg["lmstudio"]["model"]
    client = make_client(cfg)

    print(f"Endpoint: {cfg['lmstudio']['base_url']}")
    print(f"Model:    {model}")
    print(f"Prompt:   {SMOKE_PROMPT!r}")
    print()

    outputs = []
    for run in (1, 2):
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SMOKE_PROMPT}],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=SMOKE_MAX_TOKENS,
            seed=SMOKE_SEED,
        )
        elapsed = time.perf_counter() - t0
        text = response.choices[0].message.content or ""
        outputs.append(text)
        print(f"--- Run {run} ({elapsed:.2f}s) ---")
        print(text)
        print()

    if outputs[0] == outputs[1]:
        print(f"SEED HONORED: two runs with seed={SMOKE_SEED} produced identical output.")
    else:
        print(f"SEED NOT HONORED: two runs with seed={SMOKE_SEED} produced different output.")
        print("The experiment will still run, but per-prompt reproducibility is lost.")


if __name__ == "__main__":
    main()
