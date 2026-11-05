import json
from pathlib import Path

import torch

from .common import atomic_json, digest, preflight
from .data import load_index
from .features import AudioChunks, chunk_loader, prepare_features
from .model import IntelAutoencoder
from .scoring import metrics, score_recordings
from .training import checkpoint_hash


def frozen_checkpoints(output, audit, config):
    selection_path = output / "selection.json"
    if not selection_path.is_file():
        raise ValueError("TEST evaluation requires selection.json from a completed validation sweep")
    selection = json.loads(selection_path.read_text())
    expected_signature = digest(dict(config=config, data=audit["fingerprint"]))
    if selection["signature"] != expected_signature or selection["data_fingerprint"] != audit["fingerprint"]:
        raise ValueError("Frozen selection configuration/data mismatch")
    checkpoints = []
    for row in selection["runs"]:
        checkpoint = output / "runs" / row["run"] / "best.pt"
        if checkpoint_hash(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {checkpoint}")
        checkpoints.append(checkpoint)
    return checkpoints


def evaluate(config, data_root, output, device, split="TEST", checkpoint=None):
    preflight(device, config["torch_threads"])
    frame, audit = load_index(data_root)
    if split == "TEST":
        checkpoints = frozen_checkpoints(output, audit, config)
        if checkpoint is not None:
            if Path(checkpoint).resolve() not in [p.resolve() for p in checkpoints]:
                raise ValueError("TEST checkpoint is not part of the frozen selection")
            checkpoints = [Path(checkpoint)]
    elif checkpoint is not None:
        checkpoints = [Path(checkpoint)]
    else:
        raise ValueError("Specify --checkpoint for TRAIN or VAL evaluation")
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        settings = saved["settings"]
        if settings["data_fingerprint"] != audit["fingerprint"] or settings["config"] != config:
            raise ValueError("Checkpoint configuration/data mismatch")
        fft = settings["fft_size"]
        selected = frame[frame.split.eq(split)]
        cache = prepare_features(selected, data_root, output, audit, config, fft)
        dataset = AudioChunks(selected, cache, fft, config["chunk_frames"])
        model = IntelAutoencoder(fft // 2 + 1, settings["bottleneck"], config["hidden_channels"]).to(device)
        model.load_state_dict(saved["model"])
        predictions = score_recordings(model, dataset, chunk_loader(dataset, settings["batch_size"], config), device)
        directory = path.parent
        scores = directory / f"{split.lower()}_scores.csv"
        temporary = scores.with_name(scores.name + ".partial")
        predictions.to_csv(temporary, index=False)
        temporary.replace(scores)
        atomic_json(directory / f"{split.lower()}_provenance.json", dict(
            checkpoint_sha256=checkpoint_hash(path), data_fingerprint=audit["fingerprint"], split=split))
        if split != "TRAIN":
            result = metrics(predictions)
            atomic_json(directory / f"{split.lower()}_metrics.json", result)
            print(f"{directory.name} {split}: AUC={result['auc']:.5f}", flush=True)
        del model, saved
        if device == "cuda":
            torch.cuda.empty_cache()
