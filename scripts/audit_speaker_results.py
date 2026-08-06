"""Summarize speaker decisions by source and score band."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    results = json.loads(
        (ROOT / "data" / "speaker_results.json").read_text(encoding="utf-8")
    )
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    scores = defaultdict(list)
    for result in results:
        source = Path(result["path"]).name.partition("_(Vocals)")[0]
        decision = str(result.get("decision") or "missing")
        by_source[source][decision] += 1
        scores[decision].append(float(result.get("score", 0.0)))
    for source in sorted(by_source):
        print(source, dict(sorted(by_source[source].items())))
    for decision in sorted(scores):
        values = scores[decision]
        print(
            f"{decision}: {len(values)} "
            + ", ".join(
                f"p{percentile}={np.percentile(values, percentile):.4f}"
                for percentile in (1, 5, 50, 95, 99)
            )
        )


if __name__ == "__main__":
    main()
