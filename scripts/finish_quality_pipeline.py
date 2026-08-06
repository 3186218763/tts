"""Wait for speaker scoring, then finish ASR and dataset assembly."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SPEAKER_RESULTS = PROJECT_ROOT / "data" / "speaker_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--speaker-policy",
        choices=["accept", "accept-or-uncertain"],
        default="accept",
    )
    args = parser.parse_args()
    if min(
        args.poll_seconds,
        args.workers,
        args.cpu_threads_per_worker,
        args.beam_size,
    ) < 1:
        parser.error("poll and worker options must be positive")

    while not SPEAKER_RESULTS.exists():
        print("waiting for speaker_results.json", flush=True)
        time.sleep(args.poll_seconds)

    asr_command = [
        sys.executable,
        "-u",
        str(SCRIPTS_DIR / "run_parallel_asr.py"),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--cpu-threads-per-worker",
        str(args.cpu_threads_per_worker),
        "--beam-size",
        str(args.beam_size),
        "--speaker-policy",
        args.speaker_policy,
    ]
    print("speaker scoring complete; starting quality ASR", flush=True)
    subprocess.run(asr_command, cwd=PROJECT_ROOT, check=True)

    print("ASR complete; rebuilding dataset", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(SCRIPTS_DIR / "build_dataset.py"), "--stage", "dataset"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("quality dataset pipeline complete", flush=True)


if __name__ == "__main__":
    main()
