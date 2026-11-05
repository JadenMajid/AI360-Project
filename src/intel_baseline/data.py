import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

import numpy as np
import pandas as pd
import soundfile as sf

from .common import atomic_json, digest

EXPECTED = {"TRAIN": {0: 576, 1: 0}, "VAL": {0: 122, 1: 1610}, "TEST": {0: 121, 1: 1611}}
CATEGORIES = {
    "Good", "Excessive_Convexity", "Undercut", "Lack_of_Fusion", "Porosity", "Spatter",
    "Burnthrough", "Porosity_w_Excessive_Penetration", "Excessive_Penetration", "Crater_Cracks",
    "Warping", "Overlap",
}


def safe_relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in str(value):
        raise ValueError(f"Unsafe dataset path: {value}")
    return Path(*path.parts)


def cached_archive():
    base = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    repo = base / "datasets--shaundano--ai360-weld-data"
    ref = repo / "refs/main"
    if not ref.is_file():
        raise FileNotFoundError("Dataset cache not found; provide --archive or run the corrected setup script")
    revision = ref.read_text().strip()
    archive = repo / "snapshots" / revision / "intel_welding/intel_robotic_welding_dataset.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing dataset archive: {archive}")
    return archive, revision


def extract_audio(archive, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    count, seen = 0, set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts = PurePosixPath(member.name).parts
            if "intel_robotic_welding_dataset" not in parts:
                continue
            relative = safe_relative(PurePosixPath(*parts[parts.index("intel_robotic_welding_dataset") + 1:])) if parts[-1] != "intel_robotic_welding_dataset" else None
            if relative is None or (relative.name != "manifest.csv" and relative.suffix.lower() != ".flac"):
                continue
            if not member.isfile() or relative in seen:
                raise ValueError(f"Duplicate or non-regular archive member: {member.name}")
            seen.add(relative)
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Archive target escapes dataset root: {target}")
            if not target.is_file() or target.stat().st_size != member.size:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                with stream.extractfile(member) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                os.replace(temporary, target)
            if relative.suffix.lower() == ".flac":
                count += 1
                if count % 250 == 0:
                    print(f"Prepared {count} audio recordings", flush=True)
    if count != 4040 or not (root / "manifest.csv").is_file():
        raise ValueError(f"Expected 4,040 archive recordings and a manifest; found {count}")


def validate_manifest(manifest, official=True):
    required = {"CATEGORY", "SPLIT", "SUBDIRS"}
    if not required.issubset(manifest.columns) or manifest[list(required)].isna().any().any():
        raise ValueError("Manifest requires nonempty CATEGORY, SPLIT, and SUBDIRS columns")
    manifest = manifest.copy()
    for column in required:
        manifest[column] = manifest[column].astype(str).str.strip()
        if manifest[column].eq("").any():
            raise ValueError(f"Empty {column} in manifest")
    if manifest.SUBDIRS.duplicated().any():
        raise ValueError("Duplicate recording IDs in manifest")
    if not set(manifest.CATEGORY).issubset(CATEGORIES) or not set(manifest.SPLIT).issubset(EXPECTED):
        raise ValueError("Unknown category or split in manifest")
    if manifest.loc[manifest.SPLIT.eq("TRAIN"), "CATEGORY"].ne("Good").any():
        raise ValueError("Defective recordings are forbidden in TRAIN")
    for value in manifest.SUBDIRS:
        safe_relative(value)
    if official:
        actual = {s: {label: int(((manifest.SPLIT == s) & (manifest.CATEGORY.ne("Good") == bool(label))).sum())
                       for label in (0, 1)} for s in EXPECTED}
        if actual != EXPECTED or len(manifest) != 4040 or set(manifest.CATEGORY) != CATEGORIES:
            raise ValueError(f"Official split/category counts do not match: {actual}")
    return manifest


def audit_data(root, revision="provided", archive=None):
    root = Path(root).resolve()
    manifest_path = root / "manifest.csv"
    manifest = validate_manifest(pd.read_csv(manifest_path))
    records, paths = [], set()
    for position, row in enumerate(manifest.sort_values("SUBDIRS").to_dict("records"), 1):
        folder = root / safe_relative(row["SUBDIRS"])
        if not folder.resolve().is_relative_to(root) or not folder.is_dir():
            raise ValueError(f"Missing or unsafe recording directory: {folder}")
        audio = [p for p in folder.iterdir() if p.suffix.lower() in (".flac", ".wav")]
        if len(audio) != 1:
            raise ValueError(f"Expected one audio file in {folder}; found {len(audio)}")
        path = audio[0]
        if not path.resolve().is_relative_to(root) or path.resolve() in paths:
            raise ValueError(f"Duplicate or unsafe audio path: {path}")
        paths.add(path.resolve())
        waveform, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != 16000 or waveform.shape[1] != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError(f"Expected finite, nonempty, mono 16 kHz audio: {path}")
        records.append(dict(recording_id=row["SUBDIRS"], path=path.relative_to(root).as_posix(),
                            category=row["CATEGORY"], split=row["SPLIT"], label=int(row["CATEGORY"] != "Good"),
                            sample_rate=rate, n_samples=len(waveform), duration=len(waveform) / rate,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if position % 500 == 0:
            print(f"Audited {position}/{len(manifest)} recordings", flush=True)
    frame = pd.DataFrame(records)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = dict(revision=revision, manifest_sha256=manifest_hash, recordings=len(frame),
                    fingerprint=digest({"manifest": manifest_hash, "records": records}),
                    archive=str(archive) if archive else None, archive_bytes=Path(archive).stat().st_size if archive else None)
    temporary = root / "recordings.csv.partial"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, root / "recordings.csv")
    metadata["index_sha256"] = hashlib.sha256((root / "recordings.csv").read_bytes()).hexdigest()
    atomic_json(root / "audit.json", metadata)
    return metadata


def load_index(root):
    root = Path(root)
    with (root / "audit.json").open() as f:
        audit = json.load(f)
    if hashlib.sha256((root / "recordings.csv").read_bytes()).hexdigest() != audit["index_sha256"]:
        raise ValueError("Recording index changed since audit; run prepare again")
    if hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest() != audit["manifest_sha256"]:
        raise ValueError("Manifest changed since audit; run prepare again")
    frame = pd.read_csv(root / "recordings.csv")
    for row in frame.itertuples():
        path = root / safe_relative(row.path)
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or unsafe audited audio: {path}")
    return frame, audit
