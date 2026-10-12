"""The suite manifest is stable across monitoring and resume commands."""
import argparse
import tempfile
import unittest
from pathlib import Path

from run_suite import plan


class SuiteManifestTests(unittest.TestCase):
    def test_plan_reuses_existing_manifest_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                base_config=Path(__file__).resolve().parents[1] / "configs/robustness.json",
                suite="sensitivity", output=root / "output", data_root=root / "data",
                fixture=False,
            )
            manifest_path, manifest = plan(args)
            config_path = Path(manifest["conditions"][0]["config"])
            before = (manifest_path.read_bytes(), config_path.read_bytes())
            plan(args)
            self.assertEqual(before, (manifest_path.read_bytes(), config_path.read_bytes()))

            args.suite = "patterns"
            with self.assertRaisesRegex(ValueError, "conflicts"):
                plan(args)


if __name__ == "__main__":
    unittest.main()
