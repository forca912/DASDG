import unittest
import ast
from pathlib import Path
import numpy as np
from inputs import causal_inputs, assert_input_invariance, select_starts, output_markers, check_output_mapping


class ContractTests(unittest.TestCase):
    def test_input_matches_preserved_legacy_helpers_exactly(self):
        text = (Path(__file__).parent / 'vendor/train_v16b_external.py').read_text(encoding='utf-8')
        parsed = ast.parse(text)
        tree = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef)
                               and n.name in ('forward_fill', 'elapsed_steps')], type_ignores=[])
        namespace = {'np': np}
        exec(compile(tree, 'preserved_legacy_helpers', 'exec'), namespace)
        rng = np.random.default_rng(9)
        flow = rng.normal(50, 10, (100, 7)).astype(np.float32)
        mask = rng.random(flow.shape) > .4
        mask[:9, 1] = False
        normalized = (flow.astype(np.float32) - 50) / 10
        ref = np.stack([namespace['forward_fill'](normalized, mask), mask.astype(np.float32),
                        np.minimum(namespace['elapsed_steps'](mask) / 12, 4).astype(np.float32)], axis=-1)
        np.testing.assert_array_equal(causal_inputs(flow, mask, 50, 10), ref)

    def test_hidden_truth_and_future_cannot_influence_input(self):
        rng = np.random.default_rng(1)
        flow = rng.normal(30, 5, (80, 7)).astype(np.float32)
        mask = rng.random(flow.shape) > .3
        mask[:3, 0] = False
        self.assertTrue(assert_input_invariance(flow, mask, 30, 5, 40)['hidden_true_value_invariance'])

    def test_leading_missing_and_gap_are_not_backfilled(self):
        flow = np.array([[100], [300], [9], [800], [900], [12]], dtype=np.float32)
        mask = np.array([[False], [False], [True], [False], [False], [True]])
        x = causal_inputs(flow, mask, 9, 3)
        np.testing.assert_array_equal(x[:, 0, 0], [0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(x[:, 0, 2], [0, 1/12, 0, 1/12, 2/12, 0])

    def test_right_shape_but_wrong_node_time_mapping_refused(self):
        expected = output_markers(2, 5, 12)[..., None]
        # Simulate the wrong reshape often introduced during sequence decoder adaptation.
        wrong = expected.reshape(2, 5, 12, 1).transpose(0, 2, 1, 3)
        self.assertEqual(wrong.shape, expected.shape)
        with self.assertRaisesRegex(ValueError, 'identity mapping'):
            check_output_mapping(wrong, 2, 5, 12)
        check_output_mapping(expected, 2, 5, 12)

    def test_window_selection_stays_inside_split_and_no_duplicate_padding(self):
        s = select_starts(100, 50)
        self.assertEqual(len(s), 50)
        self.assertEqual((int(s[0]), int(s[-1])), (0, 76))
        self.assertTrue(np.all(s + 24 <= 100))

    def test_nonboolean_mask_refused(self):
        with self.assertRaisesRegex(ValueError, 'dtype'):
            causal_inputs(np.zeros((30, 2)), np.ones((30, 2)), 0, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
