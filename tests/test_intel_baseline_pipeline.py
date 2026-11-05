import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

from src.intel_baseline.common import DEFAULT_CONFIG, atomic_json, load_config
from src.intel_baseline.evaluation import evaluate
from src.intel_baseline.reporting import report
from src.intel_baseline.training import sweep


class PipelineTests(unittest.TestCase):
    def test_sweep_freezes_three_seeds_then_evaluates_and_reports(self):
        config = {**load_config(DEFAULT_CONFIG), "hidden_channels": 4, "epochs": 4,
                  "num_workers": 0, "wandb_mode": "disabled", "chunk_frames": 16,
                  "fft_sizes": [4], "bottlenecks": [2], "batch_sizes": [2], "seeds": [42, 43, 44]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output = root / "data", root / "artifacts"
            data.mkdir()
            output.mkdir()
            rows = []
            for i, (split, label) in enumerate([(s, y) for s in ("TRAIN", "VAL", "TEST") for y in (0, 1) if s != "TRAIN" or y == 0]):
                path = data / f"{i}.wav"
                sf.write(path, np.random.default_rng(i).normal(0, 0.1, 64), 16000, subtype="FLOAT")
                rows.append(dict(recording_id=str(i), path=path.name, split=split, label=label,
                                 category="Porosity" if label else "Good", n_samples=64,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            frame = pd.DataFrame(rows)
            audit = dict(fingerprint="fixture", recordings=len(frame), manifest_sha256="fixture")
            atomic_json(data / "audit.json", audit)
            with patch("src.intel_baseline.training.load_index", return_value=(frame, audit)), \
                 patch("src.intel_baseline.smoke.smoke", return_value={"batch_size": 2}):
                selection = sweep(config, data, output, "cpu")
                resumed = sweep(config, data, output, "cpu", resume=True)
            self.assertEqual(selection, resumed)
            self.assertEqual(len(selection["runs"]), 3)
            self.assertEqual(selection["primary"], "fft4-bn2-seed42")
            with patch("src.intel_baseline.evaluation.load_index", return_value=(frame, audit)):
                evaluate(config, data, output, "cpu")
            for row in selection["runs"]:
                scores = pd.read_csv(output / "runs" / row["run"] / "test_scores.csv")
                self.assertEqual(set(scores.recording_id.astype(str)), {"3", "4"})
            result = report(config, data, output, root / "report.md")
            self.assertIn("Test AUC across 3 seeds", result.read_text())


if __name__ == "__main__":
    unittest.main()
