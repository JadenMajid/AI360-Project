import argparse
from pathlib import Path

from .common import DEFAULT_CONFIG, ROOT, artifact_root, atomic_json, load_config


def parser():
    root = argparse.ArgumentParser(description="Intel 16 kHz acoustic welding baseline")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "train", "sweep", "evaluate", "report"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--data-root", type=Path, default=ROOT / "data/intel_baseline")
        command.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/intel_baseline")
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if name == "prepare":
            command.add_argument("--archive", type=Path)
            command.add_argument("--audit-only", action="store_true")
        if name in ("train", "sweep"):
            command.add_argument("--resume", action="store_true")
        if name == "train":
            command.add_argument("--fft-size", type=int, required=True)
            command.add_argument("--bottleneck", type=int, required=True)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--batch-size", type=int, default=16)
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path)
            command.add_argument("--split", choices=("TRAIN", "VAL", "TEST"), default="TEST")
    return root


def main():
    args = parser().parse_args()
    config = load_config(args.config)
    args.output_dir = artifact_root(args.output_dir)
    if args.command == "prepare":
        from .data import audit_data, cached_archive, extract_audio
        archive, revision = (args.archive, "provided") if args.archive else cached_archive()
        if not args.audit_only:
            extract_audio(archive, args.data_root)
        audit = audit_data(args.data_root, revision, archive)
        atomic_json(args.output_dir / "data_audit.json", audit)
        print(f"Audit passed: {audit['recordings']} recordings; fingerprint {audit['fingerprint']}")
    elif args.command == "smoke":
        from .smoke import smoke
        smoke(config, args.data_root, args.output_dir, args.device)
    elif args.command in ("train", "sweep"):
        from .common import preflight
        from .training import fit, sweep
        if args.command == "sweep":
            sweep(config, args.data_root, args.output_dir, args.device, args.resume)
        else:
            preflight(args.device, config["torch_threads"])
            if args.fft_size not in config["fft_sizes"] or args.bottleneck not in config["bottlenecks"]:
                raise ValueError("Training FFT and bottleneck must be included in the configuration")
            fit(config, args.data_root, args.output_dir, args.fft_size, args.bottleneck, args.seed,
                args.batch_size, args.device, args.resume)
    else:
        raise NotImplementedError(f"{args.command} implementation is not installed yet")


if __name__ == "__main__":
    main()
