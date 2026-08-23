"""Minimal causal forward-fill and missing-duration demonstration."""

from __future__ import annotations

import numpy as np

from dasdg.train_fdr_stgnn import elapsed_steps, forward_fill


def main() -> None:
    values = np.array([[1.0], [10.0], [20.0], [4.0]], dtype=np.float32)
    observed = np.array([[1], [0], [0], [1]], dtype=bool)
    filled = forward_fill(values, observed)
    duration = elapsed_steps(observed)
    print("values:", values.ravel().tolist())
    print("observed:", observed.astype(int).ravel().tolist())
    print("causal_forward_fill:", filled.ravel().tolist())
    print("missing_duration:", duration.ravel().tolist())


if __name__ == "__main__":
    main()
