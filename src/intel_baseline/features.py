import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .common import digest


def spectrogram(waveform, fft_size, log_floor=1e-8):
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim != 1 or not waveform.numel() or not torch.isfinite(waveform).all():
        raise ValueError("Expected finite, nonempty, mono audio")
    if waveform.numel() < fft_size:
        waveform = torch.from_numpy(np.pad(waveform.numpy(), (0, fft_size - len(waveform)), mode="reflect"))
    spectrum = torch.stft(waveform, n_fft=fft_size, hop_length=fft_size // 2,
                          window=torch.hann_window(fft_size), center=False,
                          normalized=False, onesided=True, return_complex=True)
    return spectrum.abs().clamp_min(log_floor).log().numpy()


def chunk_starts(n_frames, chunk_frames):
    if n_frames <= 0:
        raise ValueError("A recording must have at least one frame")
    if n_frames <= chunk_frames:
        return [0]
    starts = list(range(0, n_frames - chunk_frames + 1, chunk_frames))
    if starts[-1] != n_frames - chunk_frames:
        starts.append(n_frames - chunk_frames)
    return starts


def feature_path(directory, recording_id):
    return directory / (hashlib.sha256(recording_id.encode()).hexdigest() + ".npy")


def prepare_features(frame, root, output, audit, config, fft_size):
    key = digest(dict(version=1, data=audit["fingerprint"], fft=fft_size, floor=config["log_floor"]))[:20]
    directory = Path(output) / "cache" / key
    directory.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(frame.itertuples(), 1):
        path = feature_path(directory, row.recording_id)
        expected = max(1, 1 + (row.n_samples - fft_size) // (fft_size // 2))
        if path.is_file():
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            if cached.shape != (fft_size // 2 + 1, expected) or cached.dtype != np.float32:
                raise ValueError(f"Invalid feature cache: {path}; remove it and retry")
            continue
        audio = Path(root) / row.path
        if hashlib.sha256(audio.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Audio changed since audit: {audio}; run prepare --audit-only")
        waveform, rate = sf.read(audio, dtype="float32", always_2d=True)
        if rate != 16000 or waveform.shape != (row.n_samples, 1):
            raise ValueError(f"Audio shape/rate changed: {audio}")
        features = spectrogram(waveform[:, 0].copy(), fft_size, config["log_floor"])
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features: {audio}")
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as f:
            np.save(f, features, allow_pickle=False)
        os.replace(temporary, path)
        if position % 500 == 0:
            print(f"FFT {fft_size}: cached {position}/{len(frame)} recordings", flush=True)
    return directory


class AudioChunks(Dataset):
    def __init__(self, frame, directory, fft_size, chunk_frames):
        self.frame = frame.reset_index(drop=True)
        self.directory, self.chunk_frames = Path(directory), chunk_frames
        self.entries, self.n_frames = [], []
        for index, row in enumerate(self.frame.itertuples()):
            count = max(1, 1 + (row.n_samples - fft_size) // (fft_size // 2))
            self.n_frames.append(count)
            self.entries.extend((index, start, min(count, chunk_frames)) for start in chunk_starts(count, chunk_frames))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        recording, start, valid = self.entries[index]
        values = np.load(feature_path(self.directory, self.frame.iloc[recording].recording_id), mmap_mode="r", allow_pickle=False)
        chunk = values[:, start:start + self.chunk_frames].copy()
        if chunk.shape[1] < self.chunk_frames:
            chunk = np.pad(chunk, ((0, 0), (0, self.chunk_frames - chunk.shape[1])), mode="reflect")
        return torch.from_numpy(chunk), valid, recording, start


def chunk_loader(dataset, batch_size, config, shuffle=False, generator=None):
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    worker_generator = torch.Generator().manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, generator=worker_generator,
                      num_workers=config["num_workers"], pin_memory=torch.cuda.is_available(),
                      multiprocessing_context="spawn" if config["num_workers"] > 0 else None,
                      persistent_workers=config["num_workers"] > 0)
