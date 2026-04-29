"""Smoke-test the LMStudio endpoint before running the full pipeline.

Verifies:
  - The server is reachable at the configured base_url
  - A minimal chat completion returns non-empty content
  - The `enable_thinking` flag is honored (no reasoning_content when False)
  - The `seed` parameter is honored (two requests produce identical
    non-empty content)

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
SMOKE_MAX_TOKENS = 500


def main() -> None:
    cfg = load_config()
    gen_cfg = cfg["generation"]
    model = cfg["lmstudio"]["model"]
    client = make_client(cfg)

    enable_thinking = gen_cfg["enable_thinking"]

    print(f"Endpoint:        {cfg['lmstudio']['base_url']}")
    print(f"Model:           {model}")
    print(f"enable_thinking: {enable_thinking}")
    print(f"Prompt:          {SMOKE_PROMPT!r}")
    print()

    contents: list[str] = []
    reasonings: list[str] = []
    for run in (1, 2):
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SMOKE_PROMPT}],
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=SMOKE_MAX_TOKENS,
            seed=SMOKE_SEED,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        elapsed = time.perf_counter() - t0
        msg = response.choices[0].message
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        contents.append(text)
        reasonings.append(reasoning)

        print(f"--- Run {run} ({elapsed:.2f}s) ---")
        print(f"finish_reason: {response.choices[0].finish_reason}")
        if response.usage is not None:
            print(
                f"tokens: prompt={response.usage.prompt_tokens} "
                f"completion={response.usage.completion_tokens} "
                f"total={response.usage.total_tokens}"
            )
        if reasoning:
            print(f"reasoning_content ({len(reasoning)} chars):")
            print(reasoning)
        print(f"content ({len(text)} chars):")
        print(text)
        print()

    print("=== Diagnostics ===")

    # Thinking-suppression check
    if not enable_thinking:
        if any(reasonings):
            print(
                "THINKING NOT SUPPRESSED: enable_thinking=false but the server "
                "returned reasoning_content. The chat_template_kwargs hint is "
                "not being honored on this backend."
            )
        else:
            print("THINKING SUPPRESSED: no reasoning_content returned.")
    else:
        if any(reasonings):
            print("THINKING ACTIVE: reasoning_content present, as expected.")
        else:
            print(
                "WARNING: enable_thinking=true but no reasoning_content was "
                "returned. The model may not be a thinking model, or the "
                "server may not surface reasoning separately."
            )

    # Content-empty check (would make the seed test vacuous)
    if not contents[0] or not contents[1]:
        print(
            "EMPTY CONTENT: at least one run returned no visible content. "
            "Seed-honor check is skipped (would be vacuous)."
        )
        return

    # Seed-honor check, only meaningful with non-empty content
    if contents[0] == contents[1]:
        print(f"SEED HONORED: two runs with seed={SMOKE_SEED} produced identical non-empty content.")
    else:
        print(f"SEED NOT HONORED: two runs with seed={SMOKE_SEED} produced different content.")
        print("The experiment will still run, but per-prompt reproducibility is lost.")


if __name__ == "__main__":
    main()
