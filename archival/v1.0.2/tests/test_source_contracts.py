"""Dependency-light checks for the released source tree."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "dasdg"


class SourceContractTests(unittest.TestCase):
    def test_required_entrypoints_exist(self) -> None:
        required = {
            "train_fdr_stgnn.py",
            "run_dasdg_reproduction.py",
            "train_v16b_external.py",
            "run_v16b_primary.py",
            "preflight_v16b.py",
            "analyze_v16b_primary.py",
        }
        self.assertTrue(required.issubset({path.name for path in SOURCE.glob("*.py")}))

    def test_training_script_has_fixed_seed_and_independent_test_flag(self) -> None:
        text = (SOURCE / "train_v16b_external.py").read_text(encoding="utf-8")
        self.assertIn("def set_seed", text)
        self.assertIn("--independent-test", text)

    def test_orchestrator_has_validation_and_test_outputs(self) -> None:
        text = (SOURCE / "run_v16b_primary.py").read_text(encoding="utf-8")
        for token in ("validation_metrics.csv", "test_metrics.csv", "best.pt", "V16B_PRIMARY_RUN_COMPLETE"):
            self.assertIn(token, text)

    def test_preflight_checks_all_masks(self) -> None:
        text = (SOURCE / "preflight_v16b.py").read_text(encoding="utf-8")
        self.assertIn("mask_files_checked", text)
        self.assertIn("missing_mask_count", text)

    def test_orchestrators_resolve_released_source_paths(self) -> None:
        for name in ("run_dasdg_reproduction.py", "run_v16b_primary.py"):
            text = (SOURCE / name).read_text(encoding="utf-8")
            self.assertIn("SOURCE_DIR", text)
        baseline = (SOURCE / "run_v16b_primary.py").read_text(encoding="utf-8")
        self.assertIn("--data-root", baseline)
        self.assertIn('SOURCE_DIR / "train_v16b_external.py"', baseline)


if __name__ == "__main__":
    unittest.main()
