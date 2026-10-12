"""Capture a machine-readable release-validation environment record."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    record: dict[str, object] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "dasdg": package_version("dasdg"),
        },
    }
    try:
        import torch

        record["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        record["pytorch_error"] = repr(exc)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
