"""Audit cached chat candidates against downloaded M4A and WAV files."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from batch_download import probe_decodable_duration, probe_duration


def _by_bvid(directory: Path, suffix: str) -> dict[str, Path]:
    return {
        path.name.split("_", 1)[0]: path
        for path in directory.glob(f"*{suffix}")
        if "_" in path.name
    }


def _audit_one(
    candidate: dict[str, Any],
    raw_by_bvid: dict[str, Path],
    wav_by_bvid: dict[str, Path],
    training_names: set[str],
    decode: bool,
) -> dict[str, Any]:
    bvid = str(candidate["bvid"])
    expected = float(candidate.get("duration") or 0)
    raw_path = raw_by_bvid.get(bvid)
    wav_path = wav_by_bvid.get(bvid)
    record: dict[str, Any] = {
        "bvid": bvid,
        "title": candidate.get("title", ""),
        "expected_duration": expected,
        "raw_path": str(raw_path) if raw_path else None,
        "wav_path": str(wav_path) if wav_path else None,
        "raw_decodable_duration": None,
        "wav_duration": None,
        "in_training_assets": bool(wav_path and wav_path.name in training_names),
    }

    if raw_path is None:
        record["status"] = "missing_raw"
        return record
    if wav_path is None:
        record["status"] = "missing_wav"
        return record

    try:
        raw_duration = (
            probe_decodable_duration(raw_path) if decode else probe_duration(raw_path)
        )
        wav_duration = probe_duration(wav_path)
    except Exception as exc:
        record["status"] = "probe_error"
        record["error"] = str(exc)
        return record

    record["raw_decodable_duration"] = raw_duration
    record["wav_duration"] = wav_duration
    if not decode:
        record["status"] = "unchecked"
    elif expected > 0 and raw_duration < expected * 0.98:
        record["status"] = "truncated_raw"
    elif abs(wav_duration - raw_duration) > 2.0:
        record["status"] = "wav_mismatch"
    else:
        record["status"] = "ready"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decode",
        action="store_true",
        help="fully decode every M4A instead of trusting container metadata",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    candidates_path = DATA_DIR / "chat_candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    raw_by_bvid = _by_bvid(DATA_DIR / "raw", ".m4a")
    wav_by_bvid = _by_bvid(DATA_DIR / "wav", ".wav")
    training_names = {
        line.strip()
        for line in (DATA_DIR / "training_assets.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    results: list[dict[str, Any] | None] = [None] * len(candidates)
    worker_count = min(args.workers, len(candidates) or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _audit_one,
                candidate,
                raw_by_bvid,
                wav_by_bvid,
                training_names,
                args.decode,
            ): index
            for index, candidate in enumerate(candidates)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            results[index] = future.result()
            record = results[index]
            print(
                f"[{completed}/{len(candidates)}] {record['bvid']} "
                f"{record['status']}",
                flush=True,
            )

    final_results = [record for record in results if record is not None]
    output_path = DATA_DIR / "audio_asset_audit.json"
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    tmp_path.write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(output_path)

    counts: dict[str, int] = {}
    for record in final_results:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"audit: {dict(sorted(counts.items()))}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
