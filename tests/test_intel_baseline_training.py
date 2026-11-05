from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.intel_baseline.common import DEFAULT_CONFIG, load_config
from src.intel_baseline.features import AudioChunks, feature_path
from src.intel_baseline.model import IntelAutoencoder
from src.intel_baseline.training import build_optimizer, fit_datasets


class TrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        self.check_resume("adam")

    def test_muon_resume_matches_uninterrupted_training(self):
        self.check_resume("muon")

    def test_muon_updates_all_parameters_and_clears_original_gradients(self):
        torch.set_num_threads(1)
        model = IntelAutoencoder(3, bottleneck=2, hidden=4)
        config = {**load_config(DEFAULT_CONFIG), "optimizer": "muon"}
        optimizer = build_optimizer(model, config)
        originals = {name: p.detach().clone() for name, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        optimizer.step()
        for name, p in model.named_parameters():
            self.assertFalse(torch.equal(p, originals[name]), name)
        self.assertEqual(len(optimizer.muon_parameters), 8)
        optimizer.zero_grad()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        originals = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer.step()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, originals[name], rtol=0, atol=0)

    def check_resume(self, optimizer):
        torch.set_num_threads(1)
        config = {**load_config(DEFAULT_CONFIG), "hidden_channels": 4, "epochs": 4,
                  "num_workers": 2, "wandb_mode": "disabled", "chunk_frames": 16}
        if optimizer == "muon":
            config.update(optimizer="muon", muon_ns_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame([dict(recording_id="train", split="TRAIN", category="Good", label=0, n_samples=64),
                                  dict(recording_id="normal", split="VAL", category="Good", label=0, n_samples=64),
                                  dict(recording_id="defect", split="VAL", category="Porosity", label=1, n_samples=64)])
            for row in frame.itertuples():
                np.save(feature_path(root, row.recording_id), np.random.default_rng(row.Index).normal(size=(3, 31)).astype("float32"))
            train = AudioChunks(frame[frame.split.eq("TRAIN")], root, 4, 16)
            validation = AudioChunks(frame[frame.split.eq("VAL")], root, 4, 16)
            args = (train, validation, {"fingerprint": "fixture"}, 4, 2, 42, 2, "cpu")
            fit_datasets(config, root / "full", *args)
            fit_datasets(config, root / "resumed", *args, stop_after=2)
            fit_datasets(config, root / "resumed", *args, resume=True)
            full = torch.load(root / "full/runs/fft4-bn2-seed42/latest.pt", weights_only=True)
            resumed = torch.load(root / "resumed/runs/fft4-bn2-seed42/latest.pt", weights_only=True)
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
            self.assertEqual(full["scheduler"], resumed["scheduler"])
            self.assertEqual(full["best_epoch"], resumed["best_epoch"])
            self.assertEqual([r["train_mse"] for r in full["history"]], [r["train_mse"] for r in resumed["history"]])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                fit_datasets({**config, "learning_rate": 0.002}, root / "resumed", *args, resume=True)
            with patch("wandb.init", side_effect=RuntimeError("fixture network outage")):
                outage = fit_datasets({**config, "num_workers": 0, "wandb_mode": "online"}, root / "outage", *args)
            self.assertTrue(outage["complete"])
            self.assertTrue((root / "outage/runs/fft4-bn2-seed42/latest.pt").exists())
            train.frame.loc[:, "category"] = "Porosity"
            with self.assertRaisesRegex(ValueError, "Only good"):
                fit_datasets(config, root / "invalid", *args)


if __name__ == "__main__":
    unittest.main()
