import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .common import ROOT, atomic_json, digest
from .scoring import metrics

PUBLISHED = {
    "Excessive_Penetration": 0.7807, "Burnthrough": 0.7018, "Porosity": 0.9949,
    "Porosity_w_Excessive_Penetration": 0.9659, "Undercut": 0.8989, "Overlap": 0.9743,
    "Lack_of_Fusion": 0.8372, "Excessive_Convexity": 0.8842, "Spatter": 0.7796,
    "Warping": 0.7648, "Crater_Cracks": 0.9453,
}


def report(config, data_root, output, report_path=None):
    output = Path(output).resolve()
    report_path = Path(report_path) if report_path else ROOT / "reports/intel_baseline.md"
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    lines = ["# Intel audio baseline replication", "", "16 kHz adaptation; published 192 kHz reference AUC: **0.8460**.", "", "## Status", ""]
    lines += [f"Configured FFT sizes: {config['fft_sizes']}; bottlenecks: {config['bottlenecks']}; "
              f"seeds: {config['seeds']}; {config['epochs']} epochs per run.", ""]
    audit_path = Path(data_root) / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if audit:
        lines += [f"Dataset audit passed for {audit['recordings']:,} recordings.",
                  f"Manifest SHA-256: `{audit['manifest_sha256']}`.", f"Data fingerprint: `{audit['fingerprint']}`.", ""]
    else:
        lines += ["Dataset audit has not completed. No experimental results are claimed.", ""]
    grid_path = output / "validation_grid.csv"
    if grid_path.is_file():
        grid = pd.read_csv(grid_path)
        expected = len(config["fft_sizes"]) * len(config["bottlenecks"])
        lines += [f"Validation grid: {len(grid)}/{expected} configurations completed.", "",
                  "| FFT | Bottleneck | Best validation AUC | Epoch |", "|---:|---:|---:|---:|"]
        for row in grid.sort_values(["fft_size", "bottleneck"]).itertuples():
            lines.append(f"| {row.fft_size} | {row.bottleneck} | {row.best_auc:.5f} | {row.best_epoch} |")
        lines += [""]
        pivot = grid.pivot(index="fft_size", columns="bottleneck", values="best_auc")
        fig, ax = plt.subplots()
        chart = ax.imshow(pivot.values, vmin=0.5, vmax=1, aspect="auto")
        ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, xlabel="Bottleneck",
               yticks=range(len(pivot.index)), yticklabels=pivot.index, ylabel="FFT size", title="Validation ROC-AUC")
        for y in range(len(pivot.index)):
            for x in range(len(pivot.columns)):
                if np.isfinite(pivot.values[y, x]):
                    ax.text(x, y, f"{pivot.values[y, x]:.4f}", ha="center", va="center")
        fig.colorbar(chart, ax=ax)
        fig.tight_layout()
        fig.savefig(plots / "validation_grid.png")
        plt.close(fig)
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else None
    results = []
    if selection:
        if not audit or selection["signature"] != digest(dict(config=config, data=audit["fingerprint"])):
            raise ValueError("Report selection configuration/data mismatch")
        lines += [f"Frozen primary run: `{selection['primary']}`.", "", "| Run | Seed | Test AUC | Macro defect AUC |", "|---|---:|---:|---:|"]
        for row in selection["runs"]:
            directory = output / "runs" / row["run"]
            scores = directory / "test_scores.csv"
            if not scores.is_file():
                lines.append(f"| {row['run']} | {row['seed']} | pending | pending |")
                continue
            provenance = directory / "test_provenance.json"
            saved_provenance = json.loads(provenance.read_text()) if provenance.is_file() else {}
            if saved_provenance.get("checkpoint_sha256") != row["checkpoint_sha256"] or saved_provenance.get("data_fingerprint") != audit["fingerprint"]:
                raise ValueError(f"Missing or mismatched test provenance: {directory}")
            predictions = pd.read_csv(scores)
            value = metrics(predictions)
            atomic_json(directory / "test_metrics.json", value)
            results.append(dict(run=row["run"], seed=row["seed"], **value))
            lines.append(f"| {row['run']} | {row['seed']} | {value['auc']:.5f} | {value['macro_defect_auc']:.5f} |")
        lines += [""]
    else:
        lines += ["Model selection is pending. Test evaluation remains locked.", ""]
    if len(results) == len(config["seeds"]) and selection:
        aucs = [r["auc"] for r in results]
        deviation = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        lines += [f"Test AUC across {len(aucs)} seeds: **{np.mean(aucs):.5f} ± {deviation:.5f}** (sample standard deviation).", ""]
        atomic_json(output / "test_summary.json", dict(mean_auc=float(np.mean(aucs)), std_auc=deviation, runs=results))
    primary = next((r for r in results if r["run"] == selection["primary"]), None) if selection else None
    if primary:
        lines += [f"Primary test AUC: **{primary['auc']:.5f}**; difference from published reference: **{primary['auc'] - 0.846:+.5f}**.", "",
                  "| Defect | Replicated AUC | Published 192 kHz AUC |", "|---|---:|---:|"]
        for name, auc in primary["per_defect"].items():
            reference = f"{PUBLISHED[name]:.4f}" if name in PUBLISHED else "N/A"
            lines.append(f"| {name} | {auc:.5f} | {reference} |")
        lines += [""]
        scores = pd.read_csv(output / "runs" / selection["primary"] / "test_scores.csv")
        fpr, tpr, _ = roc_curve(scores.label, scores.score)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {primary['auc']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Primary test ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "test_roc.png")
        plt.close(fig)
        comparison = pd.DataFrame({"16 kHz replication": primary["per_defect"], "Published 192 kHz": PUBLISHED})
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison.plot.barh(ax=ax, xlim=(0, 1), xlabel="ROC-AUC")
        fig.tight_layout()
        fig.savefig(plots / "per_defect.png")
        plt.close(fig)
    histories = sorted((output / "runs").glob("*/history.csv"))
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for path in histories:
            history = pd.read_csv(path)
            axes[0].plot(history.epoch, history.train_mse, label=path.parent.name)
            axes[1].plot(history.epoch, history.val_auc, label=path.parent.name)
        axes[0].set(xlabel="Epoch", ylabel="Training MSE")
        axes[1].set(xlabel="Epoch", ylabel="Validation ROC-AUC")
        axes[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(plots / "training_curves.png", bbox_inches="tight")
        plt.close(fig)
    lines += ["## Figures", ""]
    for path in sorted(plots.glob("*.png")):
        lines += [f"![{path.stem.replace('_', ' ')}]({os.path.relpath(path, report_path.parent)})", ""]
    lines += ["## Checkpoints and W&B", "", f"Local artifacts: `{output}`.", ""]
    for path in sorted((output / "runs").glob("*/result.json")):
        result = json.loads(path.read_text())
        lines.append(f"- `{path.parent / 'best.pt'}`" + (f" — [W&B run]({result['wandb_url']})" if result.get("wandb_url") else " — no online run link recorded"))
    lines += ["", "## Reproduction", "", "Run from the repository root after installing requirements and authenticating W&B:", "", "```bash",
              ".venv/bin/python -m src.intel_baseline prepare",
              ".venv/bin/python -m src.intel_baseline smoke",
              ".venv/bin/python -m src.intel_baseline sweep --resume",
              ".venv/bin/python -m src.intel_baseline evaluate",
              ".venv/bin/python -m src.intel_baseline report", "```", "",
              "Use the same `--config`, `--data-root`, and `--output-dir` for all commands when overriding defaults.", "",
              "## Reproduction assumptions", "",
              "- Public mono 16 kHz audio replaces the paper's 192 kHz input; FFT lengths preserve duration, rounded to even integers.",
              "- Full recordings, channel zero, no trimming or augmentation; Hann STFT, half-window hop, no centering, natural log magnitude floored at 1e-8.",
              "- BatchNorm has no affine parameters; standardized features are the detached reconstruction target. Only training updates statistics.",
              "- Five unpadded encoder convolutions and five transposed decoder convolutions; hidden width 1,024, LeakyReLU slope 0.01, scalar output PReLU initialized to 0.25.",
              "- 192-frame chunks include an end-aligned final chunk. Short inputs are reflect-padded; padding is excluded from loss and scores, but enters convolution and training BatchNorm.",
              "- Audio shorter than one FFT window is reflect-padded to one window. An incomplete trailing STFT window is ignored.",
              "- Overlapping errors are averaged per original frame before the recording mean; higher scores indicate anomalies.",
              "- Adam, 50 epochs, peak LR 1e-4; cosine one-cycle warmup 0.3, division factors 25/10,000, fixed Adam momentum, no weight decay.",
              "- Validation selects checkpoint and configuration; test labels never influence selection or seed choice.",
              "- These choices fill omissions in the welding paper. Actual parameter counts are recorded in each run's settings.json.", "",
              "Source: [Stemmer et al., 2024](https://arxiv.org/abs/2409.02290), Tables 4, 5, 7, and 9.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Report written: {report_path}", flush=True)
    return report_path
