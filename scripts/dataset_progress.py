"""Print a compact progress report for the Huayin dataset pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _wav_hours(paths: list[Path]) -> float:
    seconds = 0.0
    for path in paths:
        try:
            with wave.open(str(path), "rb") as audio:
                seconds += audio.getnframes() / audio.getframerate()
        except (OSError, EOFError, wave.Error, ZeroDivisionError):
            continue
    return seconds / 3600


def _json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _belongs_to_assets(value: str | Path, asset_stems: set[str]) -> bool:
    name = Path(value).name
    return name.partition("_(Vocals)")[0] in asset_stems


def _active_task_pids() -> list[int]:
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            b"separate_vocals" in command
            or b"scripts.build_dataset" in command
            or b"run_parallel_separation.py" in command
            or b"batch_download.py" in command
            or b"run_parallel_asr.py" in command
        ):
            pids.append(int(entry.name))
    return sorted(pids)


def _gpu_status() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return result.stderr.strip() or "unavailable"
    return result.stdout.strip() or "no GPUs reported"


def _memory_status() -> str:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, raw_value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(raw_value.strip().split()[0])
    gib = 1024 * 1024
    return (
        f"RAM {values.get('MemAvailable', 0) / gib:.1f}/"
        f"{values.get('MemTotal', 0) / gib:.1f} GiB available; "
        f"swap {values.get('SwapFree', 0) / gib:.1f}/"
        f"{values.get('SwapTotal', 0) / gib:.1f} GiB free"
    )


def _separation_chunk_status() -> list[str]:
    statuses: list[str] = []
    for directory in sorted(Path("/tmp").glob("audio-separator-chunks-*")):
        inputs = [
            path
            for path in directory.glob("chunk_*.wav")
            if "_(" not in path.name
        ]
        outputs = list(directory.glob("chunk_*_(Vocals)*.wav"))
        statuses.append(f"{directory.name}: {len(outputs)}/{len(inputs)} chunks")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", action="store_true", help="also query NVIDIA GPU usage"
    )
    args = parser.parse_args()

    asset_names = [
        line.strip()
        for line in (DATA_DIR / "training_assets.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    asset_stems = {Path(name).stem for name in asset_names}
    asset_wavs = [DATA_DIR / "wav" / name for name in asset_names]

    vocals = sorted((DATA_DIR / "vocals").glob("*_(Vocals)*.wav"))
    selected_vocals = [
        path
        for path in vocals
        if path.name.partition("_(Vocals)")[0] in asset_stems
    ]
    slices = sorted((DATA_DIR / "slices").glob("*.wav"))
    selected_slices = [
        path
        for path in slices
        if path.name.partition("_(Vocals)")[0] in asset_stems
    ]

    all_filters = _json_list(DATA_DIR / "filter_results.json")
    filters = [
        item
        for item in all_filters
        if item.get("path") and _belongs_to_assets(item["path"], asset_stems)
    ]
    reasons = Counter(
        item.get("reason") or "keep" for item in filters
    )
    all_asr = _json_list(DATA_DIR / "asr_results.json")
    asr = [
        item
        for item in all_asr
        if item.get("path") and _belongs_to_assets(item["path"], asset_stems)
    ]
    all_speakers = _json_list(DATA_DIR / "speaker_results.json")
    speakers = [
        item
        for item in all_speakers
        if item.get("path") and _belongs_to_assets(item["path"], asset_stems)
    ]
    speaker_decisions = Counter(
        str(item.get("decision") or "missing") for item in speakers
    )
    embedding_count = 0
    embedding_cache = DATA_DIR / "speaker_embeddings.npz"
    if embedding_cache.exists():
        try:
            with np.load(embedding_cache, allow_pickle=False) as values:
                embedding_count = len(values["paths"])
        except (OSError, KeyError, ValueError):
            pass
    annotation = DATA_DIR / "dataset" / "annotation.list"
    annotation_count = (
        len(annotation.read_text(encoding="utf-8").splitlines())
        if annotation.exists()
        else 0
    )
    upstream_paths = [
        DATA_DIR / "filter_results.json",
        DATA_DIR / "speaker_results.json",
        DATA_DIR / "asr_results.json",
    ]
    annotation_stale = annotation.exists() and any(
        path.exists() and path.stat().st_mtime > annotation.stat().st_mtime
        for path in upstream_paths
    )
    candidates = _json_list(DATA_DIR / "chat_candidates.json")
    candidate_bvids = {str(item.get("bvid")) for item in candidates}
    candidate_raw = [
        path
        for path in (DATA_DIR / "raw").glob("*.m4a")
        if path.name.split("_", 1)[0] in candidate_bvids
    ]
    candidate_wavs = [
        path
        for path in (DATA_DIR / "wav").glob("*.wav")
        if path.name.split("_", 1)[0] in candidate_bvids
    ]
    part_files = list((DATA_DIR / "raw" / ".parts").rglob("*.m4a"))
    part_bytes = sum(path.stat().st_size for path in part_files if path.exists())

    print(
        f"assets: {len(asset_wavs)} files, {_wav_hours(asset_wavs):.2f} h"
    )
    print(
        f"vocals: {len(selected_vocals)}/{len(asset_wavs)} files, "
        f"{_wav_hours(selected_vocals):.2f} h"
    )
    print(
        f"slices: {len(selected_slices)} files, "
        f"{_wav_hours(selected_slices):.2f} h"
    )
    print(
        f"filter: {len(filters)} selected records "
        f"({len(all_filters) - len(filters)} legacy), {dict(sorted(reasons.items()))}"
    )
    print(
        f"speaker embeddings: {embedding_count}/{reasons.get('keep', 0)}; "
        f"decisions: {len(speakers)} {dict(sorted(speaker_decisions.items()))}"
    )
    quality_count = sum(item.get("asr_quality_version") == 1 for item in asr)
    print(
        f"asr: {len(asr)} selected records ({len(all_asr) - len(asr)} other); "
        f"quality v1 {quality_count}, legacy {len(asr) - quality_count}"
    )
    stale_suffix = " (stale)" if annotation_stale else ""
    print(f"dataset: {annotation_count} annotations{stale_suffix}")
    print(
        f"new chat downloads: raw {len(candidate_raw)}/{len(candidates)}, "
        f"wav {len(candidate_wavs)}/{len(candidates)}, "
        f"{_wav_hours(candidate_wavs):.2f} h converted; "
        f"parts {len(part_files)} files/{part_bytes / 1024**3:.2f} GiB"
    )
    print(f"active task processes: {_active_task_pids()}")
    print(f"memory: {_memory_status()}")
    chunk_status = _separation_chunk_status()
    if chunk_status:
        print("separation work: " + "; ".join(chunk_status))
    if args.gpu:
        print("gpu: index, name, memory MiB, utilization %")
        print(_gpu_status())


if __name__ == "__main__":
    main()
