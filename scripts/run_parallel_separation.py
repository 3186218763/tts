"""Use multiple GPUs to run the dataset vocal-separation stage in parallel.

Each worker receives a disjoint shard of ``data/training_assets.txt`` and
writes to the normal ``data/vocals`` directory.  The underlying pipeline is
incremental, so rerunning this script safely skips completed assets.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _import_pipeline():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import build_dataset

    return build_dataset


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def _completed(pipeline, source: Path) -> bool:
    if not pipeline.VOCALS_DIR.exists():
        return False
    prefix = f"{source.stem}_"
    source_duration = _duration(source)
    outputs = [
        path
        for path in pipeline.VOCALS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and path.name.startswith(prefix)
        and "_(Vocals)" in path.name
    ]
    return source_duration > 0 and any(
        _duration(output) >= source_duration * 0.98 for output in outputs
    )


def _balanced_shards(assets: list[Path], worker_count: int) -> list[list[Path]]:
    shards: list[list[Path]] = [[] for _ in range(worker_count)]
    totals = [0.0] * worker_count
    for asset in sorted(assets, key=_duration, reverse=True):
        index = min(range(worker_count), key=totals.__getitem__)
        shards[index].append(asset)
        totals[index] += _duration(asset)
    return shards


def _run_worker(
    worker_index: int,
    worker_count: int,
    gpu_id: str,
    asset_values: list[str],
    chunk_duration_seconds: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    pipeline = _import_pipeline()
    shard = [Path(value) for value in asset_values]
    pipeline.load_training_assets = lambda *args, **kwargs: list(shard)

    print(
        f"[worker {worker_index + 1}/{worker_count}] "
        f"GPU={gpu_id}, assets={len(shard)}",
        flush=True,
    )
    for asset in shard:
        print(f"  - {asset.name}", flush=True)
    pipeline.separate_vocals(chunk_duration_seconds=chunk_duration_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split training-assets vocal separation across multiple GPUs"
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0", "1"],
        help="CUDA device IDs, one worker per ID (default: 0 1)",
    )
    parser.add_argument(
        "--bvids",
        nargs="*",
        default=None,
        help="only process these source BVIDs",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=30,
        help="audio-separator chunk duration in minutes (default: 30)",
    )
    args = parser.parse_args()

    if not args.gpus:
        parser.error("at least one GPU ID is required")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be unique")
    if args.chunk_minutes <= 0:
        parser.error("--chunk-minutes must be positive")

    pipeline = _import_pipeline()
    assets = pipeline.load_training_assets()
    if args.bvids is not None:
        selected_bvids = set(args.bvids)
        assets = [
            asset
            for asset in assets
            if asset.name.split("_", 1)[0] in selected_bvids
        ]
        missing_bvids = selected_bvids - {
            asset.name.split("_", 1)[0] for asset in assets
        }
        if missing_bvids:
            parser.error(
                "BVIDs are not present in training_assets.txt: "
                + ", ".join(sorted(missing_bvids))
            )
    pending = [asset for asset in assets if not _completed(pipeline, asset)]
    if not pending:
        print(f"all {len(assets)} training assets are already separated")
        return

    shards = _balanced_shards(pending, min(len(args.gpus), len(pending)))
    assignments = [
        (gpu_id, shard)
        for gpu_id, shard in zip(args.gpus, shards)
        if shard
    ]
    print(
        f"pending assets: {len(pending)}, workers: {len(assignments)}",
        flush=True,
    )
    for gpu_id, shard in assignments:
        hours = sum(_duration(asset) for asset in shard) / 3600
        print(f"  GPU {gpu_id}: {len(shard)} files, {hours:.2f} h", flush=True)

    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=_run_worker,
            args=(
                index,
                len(assignments),
                gpu_id,
                [str(asset) for asset in shard],
                max(1, int(args.chunk_minutes * 60)),
            ),
            name=f"separate-gpu-{gpu_id}",
        )
        for index, (gpu_id, shard) in enumerate(assignments)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failed = [worker for worker in workers if worker.exitcode != 0]
    if failed:
        details = ", ".join(
            f"{worker.name}={worker.exitcode}" for worker in failed
        )
        raise SystemExit(f"vocal separation workers failed: {details}")


if __name__ == "__main__":
    main()
