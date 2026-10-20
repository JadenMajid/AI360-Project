"""Loaders for the pretrained audio encoders in models/ (BEATs, EAT-base, EAT-large)."""

import importlib.util
import sys
from pathlib import Path

import torch

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BEATS_DIR = Path(__file__).resolve().parent / "beats"


def load_beats(ckpt: Path = MODELS_DIR / "BEATs_iter3.pt"):
    # Microsoft's BEATs code uses top-level imports (`from backbone import ...`).
    if str(BEATS_DIR) not in sys.path:
        sys.path.insert(0, str(BEATS_DIR))
    from BEATs import BEATs, BEATsConfig

    ckpt = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_eat(name: str = "EAT-base_epoch30_pretrain"):
    # AutoModel(trust_remote_code=True) on a local folder only copies direct imports
    # of modeling_eat.py and misses model_core.py, so import the folder as a package.
    model_dir = MODELS_DIR / name
    pkg = "eat_" + name.replace("-", "_")
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg, model_dir / "__init__.py", submodule_search_locations=[str(model_dir)]
        )
        sys.modules[pkg] = importlib.util.module_from_spec(spec)
    modeling = importlib.import_module(f"{pkg}.modeling_eat")
    return modeling.EATModel.from_pretrained(model_dir).eval()
