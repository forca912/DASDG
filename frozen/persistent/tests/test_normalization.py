"""Regression tests for legacy float32 normalization; synthetic data, not PEMS evidence."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import normalization_audit, load_data, read_json, mask_path, ROOT, sha256


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(82)
        cls.features = rng.uniform(0, 600, (16000, 307, 3)).astype(np.float32)
        cls.flow = cls.features[..., 0].copy()
        cls.end = 15000
        old = cls.features[:cls.end].reshape(-1, 3)
        cls.mean = float(old.mean(axis=0)[0])
        cls.std = float(old.std(axis=0)[0])

    def test_large_legacy_reduction_passes_without_widening_tolerance(self):
        precise = self.flow[:self.end].std(dtype=np.float64)
        print('SYNTHETIC float32_std=', self.std, 'float64_std=', precise, flush=True)
        self.assertFalse(np.isclose(self.std, precise, rtol=1e-3, atol=1e-4))
        r = normalization_audit(self.flow, self.end, self.mean, self.std, self.features)
        self.assertEqual(r['matched_algorithm'], 'legacy_features_float32_axis0')
        self.assertEqual(r['saved_mean_std'], [self.mean, self.std])
        self.assertFalse(r['normalization_values_changed'])

    def test_float64_training_only_still_accepted(self):
        r = normalization_audit(self.flow, self.end, float(self.flow[:self.end].mean(dtype=np.float64)),
                               float(self.flow[:self.end].std(dtype=np.float64)), self.features)
        self.assertEqual(r['matched_algorithm'], 'flow_float64_training_only')

    def test_wrong_statistics_rejected(self):
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            normalization_audit(self.flow, self.end, self.mean, self.std*1.02, self.features)

    def test_flow_feature_mismatch_rejected(self):
        flow = self.flow.copy(); flow[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            normalization_audit(flow, self.end, self.mean, self.std, self.features)

    def test_missing_provenance_does_not_bypass_check(self):
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            normalization_audit(self.flow, self.end, self.mean, self.std)

    def test_validation_values_not_used(self):
        features = self.features.copy(); features[self.end:] += 1000
        r = normalization_audit(features[..., 0], self.end, self.mean, self.std, features)
        self.assertEqual(r['matched_algorithm'], 'legacy_features_float32_axis0')
        old = features.reshape(-1, 3)
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            normalization_audit(features[..., 0], self.end, float(old.mean(axis=0)[0]),
                                float(old.std(axis=0)[0]), features)

    def test_load_data_preserves_file_and_saved_values(self):
        with tempfile.TemporaryDirectory(prefix='SMOKE_norm_') as tmp:
            root = Path(tmp); config = read_json(ROOT/'configs/primary.json')
            rng = np.random.default_rng(4)
            features = rng.uniform(0, 600, (144, 6, 3)).astype(np.float32)
            old = features[:64].reshape(-1, 3)
            mean, std = old.mean(axis=0), old.std(axis=0)
            path = root/'processed/PEMS04/complete.npz'; path.parent.mkdir(parents=True)
            np.savez_compressed(path, features=features, flow=features[..., 0], split=[64,104,144], mean=mean, std=std)
            mp = mask_path(root, 'PEMS04', 20260808, config); mp.parent.mkdir(parents=True)
            np.savez_compressed(mp, train=np.ones((64,6),bool), val=np.ones((40,6),bool),
                                test=np.ones((40,6),bool), seed=20260808, block_length=12,
                                intended_global_missing_rate=.1, affected_node_fraction=.4, observed_is_one=True)
            before = sha256(path)
            flow, masks, got_mean, got_std, split, evidence = load_data(root,'PEMS04',20260808,config,fixture=True)
            self.assertEqual(before, sha256(path))
            self.assertEqual([got_mean, got_std], [float(mean[0]), float(std[0])])
            self.assertEqual(evidence['normalization_audit']['verification'], 'PASS')


if __name__ == '__main__':
    unittest.main(verbosity=2)
