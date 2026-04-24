"""Run the full pipeline end-to-end.

Usage:
  python run_all.py
"""
from __future__ import annotations

import subprocess
import sys


STEPS = [
    ("Generating completions via LMStudio",   "generate.py"),
    ("Embedding outputs",                     "embed.py"),
    ("Computing pairwise similarities",       "similarity.py"),
    ("Running statistical tests",             "stats.py"),
    ("Plotting",                              "plot.py"),
]


def main() -> None:
    for title, script in STEPS:
        print(f"\n=== {title} ({script}) ===")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Step failed: {script}")
            sys.exit(result.returncode)
    print("\nPipeline complete. See results/ for outputs.")


if __name__ == "__main__":
    main()
