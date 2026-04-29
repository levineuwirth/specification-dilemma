"""Generate completions for sparse and dense prompts via LMStudio.

LMStudio exposes an OpenAI-compatible server (default: localhost:1234).
Start the server from LMStudio's "Local Server" tab before running.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_client(cfg: dict) -> OpenAI:
    return OpenAI(
        base_url=cfg["lmstudio"]["base_url"],
        api_key=cfg["lmstudio"]["api_key"],
    )


def build_prompt(prompt: str, enable_thinking: bool) -> str:
    """Append Qwen3 /no_think directive when thinking is disabled.

    Using extra_body={'chat_template_kwargs': {...}} interferes with the
    seed parameter on some LMStudio backends, so we route the
    enable/disable signal through the prompt body instead.
    """
    if not enable_thinking:
        return f"{prompt}\n/no_think"
    return prompt


def generate_one(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
) -> str:
    """Single completion. Returns the assistant message content."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(prompt, enable_thinking)}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    return response.choices[0].message.content or ""


def run_condition(
    client: OpenAI,
    cfg: dict,
    condition: str,
) -> None:
    prompts_path = Path(cfg["paths"]["prompts_dir"]) / f"{condition}.json"
    outputs_dir = Path(cfg["paths"]["outputs_dir"]) / condition
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    gen_cfg = cfg["generation"]
    model = cfg["lmstudio"]["model"]

    for i, prompt in enumerate(tqdm(prompts, desc=f"{condition}")):
        out_file = outputs_dir / f"{i:02d}.txt"
        if out_file.exists():
            continue  # resume support
        text = generate_one(
            client=client,
            model=model,
            prompt=prompt,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            max_tokens=gen_cfg["max_tokens"],
            seed=i,
            enable_thinking=gen_cfg["enable_thinking"],
        )
        out_file.write_text(text, encoding="utf-8")


def main() -> None:
    cfg = load_config()
    client = make_client(cfg)
    for condition in ("sparse", "dense"):
        run_condition(client, cfg, condition)
    print("Generation complete.")


if __name__ == "__main__":
    main()
