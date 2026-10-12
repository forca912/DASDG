"""Causal input contract and identifiable output mapping fixtures (NumPy only)."""
from __future__ import annotations
import numpy as np

HISTORY = HORIZON = 12


def causal_inputs(flow, observed, mean, std):
    if flow.ndim != 2 or observed.dtype != np.bool_ or observed.shape != flow.shape:
        raise ValueError('Flow/mask shape or mask polarity dtype invalid')
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid frozen normalization')
    # Only observed entries may update the fill state. Initial fill is z-score zero.
    normalized = (flow.astype(np.float32) - mean) / std
    filled = np.zeros_like(normalized, dtype=np.float32)
    duration = np.zeros_like(normalized, dtype=np.float32)
    last = np.zeros(flow.shape[1], dtype=np.float32)
    for t in range(len(flow)):
        last[observed[t]] = normalized[t, observed[t]]
        filled[t] = last
        if t:
            duration[t] = np.where(observed[t], 0, duration[t - 1] + 1)
    duration = np.minimum(duration / HISTORY, 4).astype(np.float32)
    return np.stack([filled, observed.astype(np.float32), duration], axis=-1)


def select_starts(length, maximum):
    count = length - HISTORY - HORIZON + 1
    if count <= 0:
        raise ValueError('Segment too short')
    return np.unique(np.linspace(0, count - 1, min(count, maximum), dtype=np.int64))


def assert_input_invariance(flow, observed, mean, std, cutoff):
    before = causal_inputs(flow, observed, mean, std)[:cutoff]
    hidden_changed = flow.copy()
    hidden_changed[~observed] = 999999.0
    hidden_after = causal_inputs(hidden_changed, observed, mean, std)[:cutoff]
    future_changed = flow.copy()
    future_changed[cutoff:] = -777777.0
    future_mask = observed.copy()
    future_mask[cutoff:] = ~future_mask[cutoff:]
    future_after = causal_inputs(future_changed, future_mask, mean, std)[:cutoff]
    if not np.array_equal(before, hidden_after):
        raise ValueError('Inputs depend on hidden ground truth')
    if not np.array_equal(before, future_after):
        raise ValueError('Past input depends on future observations/mask')
    return {'cutoff_exclusive': cutoff, 'hidden_true_value_invariance': True,
            'future_value_and_mask_invariance': True,
            'normalization_statistics_held_fixed_as_training_artifact': True}


def output_markers(batch, nodes, horizon):
    # Exactly representable float32 IDs; unlike a shape probe these detect node/time swaps.
    b = np.arange(batch, dtype=np.float32)[:, None, None]
    h = np.arange(horizon, dtype=np.float32)[None, :, None]
    n = np.arange(nodes, dtype=np.float32)[None, None, :]
    return 10000 * b + 1000 * h + n


def check_output_mapping(actual, batch, nodes, horizon):
    expected = output_markers(batch, nodes, horizon)[..., None]
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError('DGCRN batch/node/horizon identity mapping failed')
    return {'batch': batch, 'nodes': nodes, 'horizons': horizon, 'marker_identity_pass': True}
