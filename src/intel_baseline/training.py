import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch

from .common import atomic_json, code_revision, digest, preflight
from .data import load_index
from .features import AudioChunks, chunk_loader, prepare_features
from .model import IntelAutoencoder, masked_mse
from .scoring import metrics, score_recordings


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state(generator):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def save_checkpoint(path, state):
    temporary = Path(path).with_suffix(".partial")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run_name(fft, bottleneck, seed):
    return f"fft{fft}-bn{bottleneck}-seed{seed}"


class RunLogger:
    def __init__(self, config, directory, metadata, resume):
        self.run, self.url = None, None
        self.path = directory / "wandb.json"
        if config["wandb_mode"] == "disabled":
            return
        try:
            import wandb
            previous = json.loads(self.path.read_text()) if self.path.is_file() else {}
            self.run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                                  name=directory.name, id=previous.get("id") if resume else None,
                                  resume="allow" if resume else None, mode=config["wandb_mode"],
                                  dir=str(directory), config=metadata,
                                  settings=wandb.Settings(init_timeout=20))
            self.url = self.run.url if config["wandb_mode"] == "online" else None
            atomic_json(self.path, dict(id=self.run.id, url=self.url, mode=config["wandb_mode"]))
        except Exception as error:
            warnings.warn(f"W&B unavailable; local logging continues: {error}")

    def log(self, values):
        if self.run is not None:
            try:
                self.run.log(values, step=values["epoch"])
            except Exception as error:
                warnings.warn(f"W&B logging interrupted; local logging continues: {error}")
                self.run = None

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                warnings.warn(f"W&B shutdown interrupted: {error}")


def make_optimizer(model, config, steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                  betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["learning_rate"], epochs=config["epochs"], steps_per_epoch=steps,
        pct_start=config["pct_start"], anneal_strategy="cos", div_factor=config["div_factor"],
        final_div_factor=config["final_div_factor"], cycle_momentum=False)
    return optimizer, scheduler


def fit(config, data_root, output, fft, bottleneck, seed, batch_size, device, resume=False):
    frame, audit = load_index(data_root)
    selected = frame[frame.split.isin(["TRAIN", "VAL"])]
    cache = prepare_features(selected, data_root, output, audit, config, fft)
    train = AudioChunks(frame[frame.split.eq("TRAIN")], cache, fft, config["chunk_frames"])
    validation = AudioChunks(frame[frame.split.eq("VAL")], cache, fft, config["chunk_frames"])
    return fit_datasets(config, output, train, validation, audit, fft, bottleneck, seed, batch_size, device, resume)


def fit_datasets(config, output, train, validation, audit, fft, bottleneck, seed, batch_size, device, resume=False, stop_after=None):
    if not train.frame.category.eq("Good").all() or not train.frame.split.eq("TRAIN").all():
        raise ValueError("Only good TRAIN recordings may be used for optimization")
    if not validation.frame.split.eq("VAL").all():
        raise ValueError("Only VAL recordings may be used for model selection")
    if set(train.frame.recording_id) & set(validation.frame.recording_id):
        raise ValueError("Training and validation recordings overlap")
    seed_everything(seed)
    directory = Path(output) / "runs" / run_name(fft, bottleneck, seed)
    directory.mkdir(parents=True, exist_ok=True)
    signature = dict(config=config, fft_size=fft, bottleneck=bottleneck, seed=seed, batch_size=batch_size,
                     data_fingerprint=audit["fingerprint"])
    signature_hash = digest(signature)
    latest = directory / "latest.pt"
    if latest.exists() and not resume:
        raise FileExistsError(f"Run already exists: {directory}; use --resume or a different output directory")
    previous = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
    if previous and previous["signature"] != signature_hash:
        raise ValueError(f"Resume configuration or data mismatch: {directory}")
    if previous and previous["epoch"] == config["epochs"]:
        result_path = directory / "result.json"
        if not result_path.exists() or not json.loads(result_path.read_text()).get("complete"):
            wandb_path = directory / "wandb.json"
            wandb = json.loads(wandb_path.read_text()) if wandb_path.exists() else {}
            atomic_json(result_path, dict(run=directory.name, fft_size=fft, bottleneck=bottleneck,
                        seed=seed, batch_size=batch_size, best_auc=previous["best_auc"],
                        best_epoch=previous["best_epoch"], epochs_completed=previous["epoch"],
                        signature=signature_hash, complete=True, wandb_url=wandb.get("url"),
                        seconds=sum(row["epoch_seconds"] for row in previous["history"])))
            pd.DataFrame(previous["history"]).to_csv(directory / "history.csv", index=False)
        return json.loads(result_path.read_text())
    generator = torch.Generator().manual_seed(seed)
    train_loader = chunk_loader(train, batch_size, config, shuffle=True, generator=generator)
    val_loader = chunk_loader(validation, batch_size, config)
    model = IntelAutoencoder(fft // 2 + 1, bottleneck, config["hidden_channels"]).to(device)
    optimizer, scheduler = make_optimizer(model, config, len(train_loader))
    first_epoch, best_auc, best_epoch = 1, -float("inf"), 0
    history = []
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        first_epoch, best_auc, best_epoch = previous["epoch"] + 1, previous["best_auc"], previous["best_epoch"]
        history = previous["history"]
        restore_rng(previous["rng"], generator)
        del previous
    metadata = dict(**signature, signature=signature_hash, revision=code_revision(),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                    torch_version=str(torch.__version__), train_chunks=len(train), validation_chunks=len(validation))
    atomic_json(directory / "settings.json", metadata)
    logger = RunLogger(config, directory, metadata, resume)
    run_start = time.monotonic()
    try:
        for epoch in range(first_epoch, config["epochs"] + 1):
            started = time.monotonic()
            model.train()
            total, denominator = 0.0, 0
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for features, lengths, _, _ in train_loader:
                features, lengths = features.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                reconstructed, target = model(features)
                loss = masked_mse(reconstructed, target, lengths)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss: {directory.name}, epoch {epoch}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                weight = int(lengths.sum()) * features.shape[1]
                total += float(loss.detach()) * weight
                denominator += weight
            training_seconds = time.monotonic() - started
            predictions = score_recordings(model, validation, val_loader, device)
            validation_auc = metrics(predictions)["auc"]
            elapsed = time.monotonic() - started
            improved = validation_auc > best_auc
            if improved:
                best_auc, best_epoch = validation_auc, epoch
            row = dict(epoch=epoch, train_mse=total / denominator, val_auc=validation_auc,
                       learning_rate=optimizer.param_groups[0]["lr"], train_seconds=training_seconds,
                       epoch_seconds=elapsed, peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0)
            history.append(row)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         rng=rng_state(generator), epoch=epoch, best_auc=best_auc, best_epoch=best_epoch,
                         history=history, settings=metadata, signature=signature_hash)
            if improved:
                save_checkpoint(directory / "best.pt", state)
                predictions.to_csv(directory / "validation_scores.csv", index=False)
            save_checkpoint(latest, state)
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            logger.log(row)
            print(f"{directory.name}: epoch {epoch}/{config['epochs']}, MSE={row['train_mse']:.5f}, "
                  f"val AUC={validation_auc:.5f}, {elapsed:.1f}s", flush=True)
            if epoch == 1:
                atomic_json(directory / "benchmark.json", dict(epoch_seconds=elapsed, training_seconds=training_seconds,
                            estimated_run_seconds=elapsed * config["epochs"], batch_size=batch_size))
                estimated = elapsed * config["epochs"] * (len(config["fft_sizes"]) * len(config["bottlenecks"]) + len(config["seeds"]) - 1)
                atomic_json(Path(output) / "sweep_estimate.json", dict(estimated_seconds=estimated,
                            basis=directory.name, note="Extrapolation; FFTs have different chunk counts"))
                print(f"Estimated run: {elapsed * config['epochs'] / 3600:.2f}h; "
                      f"grid plus repeats extrapolation: {estimated / 3600:.2f}h", flush=True)
            result = dict(run=directory.name, fft_size=fft, bottleneck=bottleneck, seed=seed, batch_size=batch_size,
                          best_auc=best_auc, best_epoch=best_epoch, epochs_completed=epoch, signature=signature_hash,
                          complete=epoch == config["epochs"], wandb_url=logger.url,
                          seconds=time.monotonic() - run_start)
            atomic_json(directory / "result.json", result)
            if stop_after is not None and epoch >= stop_after:
                break
    finally:
        logger.finish()
        del train_loader, val_loader, model, optimizer, scheduler
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return result


def checkpoint_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sweep(config, data_root, output, device, resume=False):
    preflight(device, config["torch_threads"])
    from .smoke import smoke
    _, audit = load_index(data_root)
    state_path = output / "sweep.json"
    signature = digest(dict(config=config, data=audit["fingerprint"]))
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not resume:
            raise FileExistsError("Sweep already exists; use --resume or a different output directory")
        if state["signature"] != signature:
            raise ValueError("Sweep configuration/data changed; use a different output directory")
    else:
        smoke_result = smoke(config, data_root, output, device)
        state = dict(signature=signature, batch_size=smoke_result["batch_size"], data_fingerprint=audit["fingerprint"], config=config)
        atomic_json(state_path, state)
    batch, results = state["batch_size"], []
    for fft in sorted(config["fft_sizes"]):
        for bottleneck in sorted(config["bottlenecks"]):
            results.append(fit(config, data_root, output, fft, bottleneck, config["seeds"][0], batch, device, resume))
            pd.DataFrame(results).to_csv(output / "validation_grid.csv", index=False)
            if len(results) == 1:
                benchmark = json.loads((output / "runs" / results[0]["run"] / "benchmark.json").read_text())
                estimate = benchmark["epoch_seconds"] * config["epochs"] * (len(config["fft_sizes"]) * len(config["bottlenecks"]) + len(config["seeds"]) - 1)
                atomic_json(output / "sweep_estimate.json", dict(estimated_seconds=estimate,
                            note="First FFT/configuration estimate; later FFTs have different chunk counts"))
    best = sorted(results, key=lambda r: (-r["best_auc"], r["fft_size"], r["bottleneck"]))[0]
    winners = [best]
    for seed in config["seeds"][1:]:
        winners.append(fit(config, data_root, output, best["fft_size"], best["bottleneck"], seed, batch, device, resume))
    selection = dict(signature=signature, data_fingerprint=audit["fingerprint"], primary=best["run"],
                     fft_size=best["fft_size"], bottleneck=best["bottleneck"],
                     runs=[dict(run=r["run"], seed=r["seed"], best_epoch=r["best_epoch"],
                                checkpoint_sha256=checkpoint_hash(output / "runs" / r["run"] / "best.pt")) for r in winners])
    if (output / "selection.json").exists() and json.loads((output / "selection.json").read_text()) != selection:
        raise ValueError("Frozen selection differs; use a new output directory")
    atomic_json(output / "selection.json", selection)
    print(f"Selection frozen: {best['run']}; test evaluation is now enabled", flush=True)
    return selection
