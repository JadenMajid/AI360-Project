import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.intel_baseline.common import DEFAULT_CONFIG, atomic_json, digest, load_config
from src.intel_baseline.evaluation import frozen_checkpoints
from src.intel_baseline.reporting import report
from src.intel_baseline.training import checkpoint_hash


class ReportingTests(unittest.TestCase):
    def test_freeze_gate_and_prediction_only_report(self):
        config = {**load_config(DEFAULT_CONFIG), "fft_sizes": [342], "bottlenecks": [16], "seeds": [42]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output = root / "data", root / "artifacts"
            data.mkdir()
            output.mkdir()
            audit = dict(fingerprint="fixture", manifest_sha256="fixture", recordings=4)
            atomic_json(data / "audit.json", audit)
            with self.assertRaisesRegex(ValueError, "completed validation sweep"):
                frozen_checkpoints(output, audit, config)
            run = "fft342-bn16-seed42"
            run_dir = output / "runs" / run
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best.pt"
            checkpoint.write_bytes(b"fixture")
            checksum = checkpoint_hash(checkpoint)
            selection = dict(signature=digest(dict(config=config, data="fixture")), data_fingerprint="fixture",
                             primary=run, runs=[dict(run=run, seed=42, checkpoint_sha256=checksum)])
            atomic_json(output / "selection.json", selection)
            self.assertEqual(frozen_checkpoints(output, audit, config), [checkpoint])
            pd.DataFrame(dict(recording_id=list("abcd"), split=["TEST"] * 4, label=[0, 0, 1, 1],
                              category=["Good", "Good", "Porosity", "Spatter"], score=[0.1, 0.2, 0.3, 0.4],
                              n_frames=[100] * 4)).to_csv(run_dir / "test_scores.csv", index=False)
            atomic_json(run_dir / "test_provenance.json", dict(checkpoint_sha256=checksum, data_fingerprint="fixture"))
            path = report(config, data, output, root / "report.md")
            self.assertIn("Primary test AUC: **1.00000**", path.read_text())
            self.assertEqual(json.loads((output / "test_summary.json").read_text())["mean_auc"], 1)
            self.assertTrue((output / "plots/test_roc.png").exists())
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                frozen_checkpoints(output, audit, config)


if __name__ == "__main__":
    unittest.main()
