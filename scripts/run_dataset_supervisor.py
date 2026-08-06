"""Wait for vocal separation, then finish the dataset pipeline automatically."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_dataset as pipeline


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def _vocal_output(source: Path) -> Path | None:
    if not pipeline.VOCALS_DIR.exists():
        return None
    prefix = f"{source.stem}_"
    outputs = [
        path
        for path in pipeline.VOCALS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".wav"
        and path.name.startswith(prefix)
        and "_(Vocals)" in path.name
    ]
    return max(outputs, key=_wav_duration, default=None)


def _completed_vocals(assets: list[Path]) -> tuple[list[Path], list[Path]]:
    complete: list[Path] = []
    pending: list[Path] = []
    for source in assets:
        output = _vocal_output(source)
        source_duration = _wav_duration(source)
        output_duration = _wav_duration(output) if output else 0.0
        if source_duration > 0 and output_duration >= source_duration * 0.98:
            complete.append(source)
        else:
            pending.append(source)
    return complete, pending


def _wait_for_vocals(assets: list[Path], poll_seconds: int) -> None:
    last_complete = -1
    last_heartbeat = 0.0
    while True:
        complete, pending = _completed_vocals(assets)
        now = time.monotonic()
        if len(complete) != last_complete or now - last_heartbeat >= 600:
            print(
                f"vocal separation: {len(complete)}/{len(complete) + len(pending)} complete",
                flush=True,
            )
            for source in pending:
                print(f"  waiting: {source.name}", flush=True)
            last_complete = len(complete)
            last_heartbeat = now
        if not pending:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--slice-workers", type=int, default=4)
    parser.add_argument("--filter-workers", type=int, default=12)
    parser.add_argument("--speaker-workers", type=int, default=4)
    parser.add_argument("--speaker-threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--speaker-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="声纹 embedding 设备（默认 auto → CUDA）",
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument(
        "--speaker-policy",
        choices=["accept", "accept-or-uncertain"],
        default="accept",
        help="ASR 声纹策略：默认只转 accept，保证训练集纯度",
    )
    args = parser.parse_args()

    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if min(
        args.slice_workers,
        args.filter_workers,
        args.speaker_workers,
        args.speaker_threads_per_worker,
    ) < 1:
        parser.error("worker counts must be positive")

    load_assets = pipeline.load_training_assets
    frozen_assets = load_assets()
    expected_assets = tuple(path.name for path in frozen_assets)
    # Keep every in-process stage on the same immutable batch even if another
    # process appends a newly downloaded recording to the whitelist.
    pipeline.load_training_assets = lambda *unused, **also_unused: list(frozen_assets)

    def assert_assets_unchanged() -> None:
        current = tuple(path.name for path in load_assets())
        if current != expected_assets:
            raise SystemExit("training whitelist changed during supervised batch")

    _wait_for_vocals(frozen_assets, args.poll_seconds)
    print("all vocals are complete; allowing writers to close", flush=True)
    time.sleep(30)

    assert_assets_unchanged()
    print("starting parallel slice stage", flush=True)
    pipeline.slice_audio(workers=args.slice_workers)

    assert_assets_unchanged()
    print("starting parallel filter stage", flush=True)
    pipeline.filter_slices(workers=args.filter_workers)

    assert_assets_unchanged()
    print(
        f"starting speaker verification stage (device={args.speaker_device})",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(SCRIPTS_DIR / "filter_speakers.py"),
            "--device",
            args.speaker_device,
            "--gpus",
            *args.gpus,
            "--workers",
            str(args.speaker_workers),
            "--threads-per-worker",
            str(args.speaker_threads_per_worker),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    assert_assets_unchanged()
    print(
        f"starting multi-GPU ASR stage (speaker_policy={args.speaker_policy})",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(SCRIPTS_DIR / "run_parallel_asr.py"),
            "--gpus",
            *args.gpus,
            "--whisper-model",
            args.whisper_model,
            "--speaker-policy",
            args.speaker_policy,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("building final dataset", flush=True)
    pipeline.build_dataset()
    print("dataset pipeline complete", flush=True)


if __name__ == "__main__":
    main()
