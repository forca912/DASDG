"""Unit tests for R1.4 mask provenance and pattern behavior."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_masks import generate_one, _select_nodes, _make_split
from protocol import mask_path, read_json


class MaskGenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs/robustness.json")
        self.config["datasets"] = ["PEMS04"]
        self.config["seeds"] = [7]
        self.config["global_missing_rate"] = 0.2
        self.config["affected_node_fraction"] = 0.5
        self.config["block_length"] = 6

    def _data(self, root: Path, changed_tail: bool = False):
        rng = np.random.default_rng(123)
        flow = rng.normal(size=(240, 8)).astype(np.float32)
        if changed_tail:
            flow[160:] += 10000.0
        path = root / "processed/PEMS04/complete.npz"
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, flow=flow, split=[144, 192, 240],
                            mean=[float(flow[:144].mean())], std=[float(flow[:144].std())])

    def test_metadata_and_rate_are_auditable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._data(root)
            item = generate_one(root, "PEMS04", 7, self.config)
            path = Path(item["mask_path"])
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(str(z["mask_pattern"].item()), "regular")
                self.assertEqual(str(z["mask_generator_version"].item()), "r1-robustness-mask-v1")
                self.assertTrue(bool(z["observed_is_one"]))
                self.assertEqual(z["selected_nodes"].tolist(), item["selected_nodes"])
                for name in ("train", "val", "test"):
                    self.assertAlmostEqual(float((~z[name]).mean()), 0.2, delta=1 / z[name].size)

    def test_irregular_duration_has_non_constant_run_lengths(self):
        config = copy.deepcopy(self.config)
        config["mask_pattern"] = "irregular_duration"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._data(root)
            item = generate_one(root, "PEMS04", 7, config)
            with np.load(item["mask_path"], allow_pickle=False) as z:
                lengths = []
                for name in ("train", "val", "test"):
                    for node in range(z[name].shape[1]):
                        missing = np.r_[False, ~z[name][:, node], False].astype(np.int8)
                        changes = np.diff(missing)
                        lengths.extend((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist())
            self.assertGreater(len(set(lengths)), 1)

    def test_spatial_selection_uses_training_only(self):
        config = copy.deepcopy(self.config)
        config["mask_pattern"] = "spatial_cluster"
        with tempfile.TemporaryDirectory() as tmp:
            root_a, root_b = Path(tmp) / "a", Path(tmp) / "b"
            self._data(root_a)
            self._data(root_b, changed_tail=True)
            a = generate_one(root_a, "PEMS04", 7, config)
            b = generate_one(root_b, "PEMS04", 7, config)
            self.assertEqual(a["selected_nodes"], b["selected_nodes"])
            self.assertEqual(a["cluster_basis"], "training_only_absolute_correlation_proxy")

    def test_spatial_unique_nodes_and_rates_at_real_node_counts(self):
        for nodes in (307, 170):
            flow = np.random.default_rng(123).normal(size=(60, nodes)).astype(np.float32)
            for fraction in (1 / nodes, 0.4, 1.0):
                with self.subTest(nodes=nodes, fraction=fraction):
                    selected, meta = _select_nodes(flow, 36, fraction, 20260808, "spatial_cluster")
                    count = max(1, int(round(nodes * fraction)))
                    self.assertEqual(len(selected), count)
                    self.assertEqual(len(np.unique(selected)), count)
                    self.assertIn(meta["cluster_anchor"], selected)
                    rate = min(0.1, fraction / 2)
                    observed = _make_split(120, nodes, selected, rate, 12, "spatial_cluster",
                                           np.random.default_rng(12))
                    actual_nodes = np.flatnonzero((~observed).any(axis=0))
                    np.testing.assert_array_equal(actual_nodes, selected)
                    self.assertAlmostEqual(float((~observed).mean()), rate, delta=1 / observed.size)
                    self.assertLessEqual(abs(len(actual_nodes) / nodes - fraction), 1 / nodes)

    def test_path_separates_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._data(root)
            regular = generate_one(root, "PEMS04", 7, self.config)
            config = copy.deepcopy(self.config)
            config["mask_pattern"] = "spatial_cluster"
            spatial = generate_one(root, "PEMS04", 7, config)
            self.assertNotEqual(regular["mask_path"], spatial["mask_path"])
            self.assertNotEqual(mask_path(root, "PEMS04", 7, self.config),
                                mask_path(root, "PEMS04", 7, config))


if __name__ == "__main__":
    unittest.main(verbosity=2)
