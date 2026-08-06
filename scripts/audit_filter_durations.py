"""Report duration and feature distributions for acoustic-filtered slices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    records = json.loads((ROOT / "data" / "filter_results.json").read_text(encoding="utf-8"))
    kept = [record for record in records if record.get("keep") is True]
    for key in ("duration", "rms_db", "voiced_ratio", "longest_voiced"):
        values = [float(record[key]) for record in kept if record.get(key) is not None]
        print(
            f"{key}: "
            + ", ".join(f"p{p}={np.percentile(values, p):.3f}" for p in (1, 5, 25, 50, 75, 95, 99))
        )


if __name__ == "__main__":
    main()
