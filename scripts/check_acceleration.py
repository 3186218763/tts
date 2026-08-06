"""Report the acceleration providers available to the Huayin runtime."""

from __future__ import annotations

import importlib.util
import os


def main() -> None:
    print(f"cpu.logical_count: {os.cpu_count()}")
    try:
        print(f"cpu.loadavg: {', '.join(f'{value:.2f}' for value in os.getloadavg())}")
    except OSError:
        pass
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    except Exception as exc:
        print(f"torch: unavailable ({exc})")

    try:
        import onnxruntime as ort

        print(f"onnxruntime: {ort.__version__}")
        print(f"onnxruntime.providers: {ort.get_available_providers()}")
    except Exception as exc:
        print(f"onnxruntime: unavailable ({exc})")

    for package in ("audio_separator", "faster_whisper", "librosa"):
        print(f"{package}: {'installed' if importlib.util.find_spec(package) else 'missing'}")


if __name__ == "__main__":
    main()
