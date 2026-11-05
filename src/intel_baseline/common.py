import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/intel_baseline.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def load_config(path):
    with DEFAULT_CONFIG.open() as f:
        config = json.load(f)
    with Path(path).open() as f:
        supplied = json.load(f)
    unknown = set(supplied) - set(config)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config.update(supplied)
    for key in ("sample_rate", "hidden_channels", "chunk_frames", "epochs", "torch_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("fft_sizes", "bottlenecks", "seeds", "batch_sizes"):
        values = config[key]
        if not values or any(type(v) is not int or v < (0 if key == "seeds" else 1) for v in values):
            raise ValueError(f"Invalid {key}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {key}")
    if config["sample_rate"] != 16000 or config["chunk_frames"] < 11:
        raise ValueError("Use 16 kHz audio and at least eleven frames per chunk")
    if any(n < 2 or n % 2 for n in config["fft_sizes"]):
        raise ValueError("FFT sizes must be positive even integers")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if config["wandb_mode"] not in ("online", "offline", "disabled"):
        raise ValueError("wandb_mode must be online, offline, or disabled")
    for key in ("learning_rate", "adam_epsilon", "log_floor", "div_factor", "final_div_factor"):
        if not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < config["pct_start"] < 1 or config["weight_decay"] < 0:
        raise ValueError("Invalid scheduler or weight decay configuration")
    if len(config["adam_betas"]) != 2 or any(not 0 <= b < 1 for b in config["adam_betas"]):
        raise ValueError("Invalid Adam betas")
    return config


def code_revision():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}


def artifact_root(path):
    path = Path(path).resolve()
    git_entry = ROOT / ".git"
    if git_entry.is_file() and path.is_relative_to(ROOT):
        raise ValueError("Store artifacts outside this worktree using --output-dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def preflight(device, threads=4):
    import importlib.metadata
    import torch
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run on the GPU host outside the restricted sandbox; CPU is only for development checks")
    selected = torch.device(device)
    layer = torch.nn.Conv1d(3, 4, 3).to(selected)
    layer(torch.ones(2, 3, 12, device=selected)).square().mean().backward()
    if not all(torch.isfinite(p.grad).all() for p in layer.parameters()):
        raise RuntimeError("Non-finite CUDA preflight gradients")
    return dict(device=str(selected), gpu=torch.cuda.get_device_name(selected) if selected.type == "cuda" else None,
                versions={name: importlib.metadata.version(name) for name in
                          ("torch", "numpy", "pandas", "soundfile", "scikit-learn", "wandb", "matplotlib")})
