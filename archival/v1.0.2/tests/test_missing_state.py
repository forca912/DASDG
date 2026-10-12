"""Core missing-state representation tests."""

import numpy as np
import unittest

from dasdg.train_fdr_stgnn import elapsed_steps, forward_fill


class MissingStateTests(unittest.TestCase):
    def test_elapsed_duration_resets_after_observation(self) -> None:
        observed = np.array([[1], [0], [0], [1], [0]], dtype=bool)
        expected = np.array([[0], [1], [2], [0], [1]], dtype=np.float32)
        np.testing.assert_array_equal(elapsed_steps(observed), expected)

    def test_forward_fill_uses_only_current_and_past_values(self) -> None:
        values = np.array([[1.0], [10.0], [20.0], [4.0]], dtype=np.float32)
        observed = np.array([[1], [0], [0], [1]], dtype=bool)
        expected = np.array([[1.0], [1.0], [1.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_equal(forward_fill(values, observed), expected)


if __name__ == "__main__":
    unittest.main()
