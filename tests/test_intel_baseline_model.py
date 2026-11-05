from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.intel_baseline.features import AudioChunks, chunk_starts, feature_path, spectrogram
from src.intel_baseline.model import IntelAutoencoder, masked_mse
from src.intel_baseline.scoring import metrics, score_recordings


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fft_shapes_and_short_audio(self):
        for fft in (342, 1366, 2730, 5462):
            self.assertEqual(spectrogram(np.ones(fft * 2), fft).shape, (fft // 2 + 1, 3))
            self.assertTrue(np.isfinite(spectrogram(np.ones(1), fft)).all())

    def test_topology_gradients_and_mask(self):
        model = IntelAutoencoder(9, 3, hidden=8)
        features = torch.randn(2, 9, 23)
        self.assertEqual(model.encoder(model.standardizer(features)).shape, (2, 3, 13))
        reconstructed, target = model(features)
        self.assertEqual(reconstructed.shape, features.shape)
        loss = masked_mse(reconstructed, target, torch.tensor([23, 11]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        a = torch.zeros(1, 2, 12)
        b = a.clone()
        b[:, :, 11] = 100
        self.assertEqual(float(masked_mse(a, b, torch.tensor([11]))), 0)

    def test_batch_independent_scoring_bn_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frame = pd.DataFrame([dict(recording_id="a", split="VAL", category="Good", label=0, n_samples=104),
                                  dict(recording_id="b", split="VAL", category="Porosity", label=1, n_samples=12)])
            for row in frame.itertuples():
                np.save(feature_path(path, row.recording_id), np.random.default_rng(row.Index).normal(size=(3, max(1, (row.n_samples - 4) // 2 + 1))).astype("float32"))
            dataset = AudioChunks(frame, path, 4, 16)
            model = IntelAutoencoder(3, 2, hidden=4)
            model(torch.randn(2, 3, 16))
            mean, variance = model.standardizer.running_mean.clone(), model.standardizer.running_var.clone()
            first = score_recordings(model, dataset, DataLoader(dataset, batch_size=1), "cpu")
            second = score_recordings(model, dataset, DataLoader(dataset, batch_size=3), "cpu")
            np.testing.assert_allclose(first.score, second.score, rtol=1e-5)
            torch.testing.assert_close(model.standardizer.running_mean, mean)
            torch.testing.assert_close(model.standardizer.running_var, variance)
            torch.save(model.state_dict(), path / "model.pt")
            restored = IntelAutoencoder(3, 2, hidden=4)
            restored.load_state_dict(torch.load(path / "model.pt", weights_only=True))
            third = score_recordings(restored, dataset, DataLoader(dataset, batch_size=1), "cpu")
            np.testing.assert_allclose(first.score, third.score)
            self.assertEqual(chunk_starts(35, 16), [0, 16, 19])

    def test_auc_direction_and_per_defect_negatives(self):
        frame = pd.DataFrame(dict(recording_id=list("abcd"), label=[0, 0, 1, 1],
                                  category=["Good", "Good", "Porosity", "Spatter"], score=[0.1, 0.2, 0.3, 0.4]))
        result = metrics(frame)
        self.assertEqual(result["auc"], 1)
        self.assertEqual(result["per_defect"], {"Porosity": 1, "Spatter": 1})
        frame.score *= -1
        self.assertEqual(metrics(frame)["auc"], 0)


if __name__ == "__main__":
    unittest.main()
