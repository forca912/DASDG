"""Link the persistent v15 data directory into the v16A package safely."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/autodl-tmp/traffic_missing_forecasting_tools_v15_confirm/data"),
    )
    parser.add_argument("--target", type=Path, default=Path("data"))
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.target
    if not source.exists():
        raise FileNotFoundError(source)
    if target.is_symlink():
        current = target.resolve()
        if current != source:
            raise RuntimeError(f"Existing data link points to {current}, expected {source}")
        print(f"data link already correct: {target} -> {source}")
        return
    if target.exists():
        raise RuntimeError(f"Refusing to replace existing non-symlink path: {target}")
    os.symlink(source, target, target_is_directory=True)
    print(f"created data link: {target} -> {source}")


if __name__ == "__main__":
    main()
