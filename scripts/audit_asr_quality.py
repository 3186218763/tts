"""Summarize cached faster-whisper quality metadata without changing data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASR_PATH = PROJECT_ROOT / "data" / "asr_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=ASR_PATH)
    args = parser.parse_args()
    records = json.loads(args.path.read_text(encoding="utf-8"))
    versioned = [
        record for record in records if record.get("asr_quality_version") == 1
    ]
    print(f"records: {len(records)}, quality-v1: {len(versioned)}")
    print("languages:", dict(Counter(str(r.get("lang") or "") for r in versioned)))
    for key in (
        "language_probability",
        "avg_logprob",
        "max_no_speech_probability",
        "max_compression_ratio",
    ):
        values = [float(r[key]) for r in versioned if r.get(key) is not None]
        if not values:
            print(f"{key}: no values")
            continue
        print(
            f"{key}: "
            + ", ".join(
                f"p{p}={np.percentile(values, p):.4f}"
                for p in (1, 5, 25, 50, 75, 95, 99)
            )
        )


if __name__ == "__main__":
    main()
