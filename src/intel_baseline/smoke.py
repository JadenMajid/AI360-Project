import gc
import time

import numpy as np
import torch

from .common import atomic_json, preflight
from .data import load_index
from .features import AudioChunks, chunk_loader, prepare_features, spectrogram
from .model import IntelAutoencoder, masked_mse
from .scoring import score_recordings


def choose_batch(config, device):
    fft, bottleneck = max(config["fft_sizes"]), max(config["bottlenecks"])
    for batch_size in config["batch_sizes"]:
        model = optimizer = features = loss = output = target = None
        try:
            model = IntelAutoencoder(fft // 2 + 1, bottleneck, config["hidden_channels"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            features = torch.randn(batch_size, fft // 2 + 1, config["chunk_frames"], device=device)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output, target = model(features)
                loss = masked_mse(output, target, torch.full((batch_size,), config["chunk_frames"], device=device))
                loss.backward()
                optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite batch preflight loss")
            return batch_size
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size} exhausted GPU memory; trying the next size", flush=True)
        finally:
            del model, optimizer, features, loss, output, target
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("No configured batch size fits the GPU")


def smoke(config, data_root, output_dir, device):
    environment = preflight(device, config["torch_threads"])
    batch = choose_batch(config, device)
    torch.manual_seed(config["seeds"][0])
    for fft in config["fft_sizes"]:
        assert spectrogram(np.ones(fft * 2, dtype=np.float32), fft).shape == (fft // 2 + 1, 3)
    frame, audit = load_index(data_root)
    selected = frame[frame.split.eq("TRAIN")].head(2)
    fft = config["fft_sizes"][1] if len(config["fft_sizes"]) > 1 else config["fft_sizes"][0]
    cache = prepare_features(selected, data_root, output_dir, audit, config, fft)
    dataset = AudioChunks(selected, cache, fft, config["chunk_frames"])
    loader_config = {**config, "num_workers": 0}
    loader = chunk_loader(dataset, min(batch, 2), loader_config)
    features, lengths, _, _ = next(iter(loader))
    features, lengths = features.to(device), lengths.to(device)
    model = IntelAutoencoder(fft // 2 + 1, 48, config["hidden_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    losses, started = [], time.monotonic()
    for _ in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstructed, target = model(features)
        loss = masked_mse(reconstructed, target, lengths)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite real-data smoke loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if losses[-1] >= losses[0] * 0.8:
        raise RuntimeError(f"Real-data overfit failed: {losses[0]:.5f} -> {losses[-1]:.5f}")
    before = score_recordings(model, dataset, loader, device)
    checkpoint = output_dir / "smoke.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = IntelAutoencoder(fft // 2 + 1, 48, config["hidden_channels"]).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    after = score_recordings(restored, dataset, loader, device)
    np.testing.assert_allclose(before.score, after.score, rtol=1e-6, atol=1e-7)
    result = dict(environment=environment, batch_size=batch, initial_mse=losses[0], final_mse=losses[-1],
                  steps=len(losses), seconds=time.monotonic() - started, data_fingerprint=audit["fingerprint"],
                  parameters=sum(p.numel() for p in model.parameters()), fft_size=fft, checkpoint=str(checkpoint))
    atomic_json(output_dir / "smoke.json", result)
    print(f"Smoke passed on {environment['gpu'] or device}: batch={batch}, MSE {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    return result
