# Intel audio baseline

This pipeline adapts the Intel 1D CNN autoencoder to the released mono 16 kHz audio. The reference test AUC of 0.846 used 192 kHz audio. The implementation assumptions are recorded in the generated report.

## Run the milestone

Finish the dataset download first. The preparer rejects active Hugging Face download locks, incomplete cache blobs, and an archive that changes during extraction. It uses the cached `shaundano/ai360-weld-data` archive and extracts only the manifest and audio.

From the repository root, with the virtual environment installed and W&B authenticated:

```bash
bash scripts/run_intel_baseline.sh
```

This runs preparation, the GPU smoke/overfit check, all 16 validation configurations, two repeat seeds of the winner, frozen-selection test evaluation, and report generation. Existing runs resume from their latest completed epoch. No dataset download is initiated by the runner.

The full experiment uses 50 epochs per run, width 1,024, float32, four FFT durations, and four bottleneck sizes. It requires a CUDA host. The first complete epoch produces a runtime estimate; estimates vary with FFT size. Feature caches can consume tens of GB. Checkpoints include optimizer and RNG state, so reserve additional disk space for the 18 runs.

Override locations through `INTEL_BASELINE_CONFIG`, `INTEL_BASELINE_DATA_ROOT`, `INTEL_BASELINE_OUTPUT_DIR`, or `INTEL_BASELINE_PYTHON`. Keep output outside worktrees. The launcher records the effective configuration in each run's settings and W&B.

## Individual commands

```bash
.venv/bin/python -m src.intel_baseline prepare
.venv/bin/python -m src.intel_baseline smoke
.venv/bin/python -m src.intel_baseline train --fft-size 1366 --bottleneck 48 --seed 42
.venv/bin/python -m src.intel_baseline sweep --resume
.venv/bin/python -m src.intel_baseline evaluate
.venv/bin/python -m src.intel_baseline report
```

All commands accept `--config`, `--data-root`, and `--output-dir`. Use the same values throughout an experiment. `prepare --archive PATH` selects another archive; `prepare --audit-only` audits an existing extraction without requiring a cached download. `--device cpu` is intended for development checks.

`train --resume` requires identical settings and data. Its configuration must be included in the configured grid. A manually trained grid member can subsequently be reused by `sweep --resume` if its seed, batch size, and full configuration agree.

`evaluate` defaults to TEST and requires the completed sweep's `selection.json`. Without `--checkpoint`, it evaluates all selected seeds. A specified test checkpoint must match the frozen selection's hash. For exploratory validation scoring:

```bash
.venv/bin/python -m src.intel_baseline evaluate --split VAL \
  --checkpoint artifacts/intel_baseline/runs/fft1366-bn48-seed42/best.pt
```

## Artifacts

- `data/intel_baseline/`: official manifest, FLAC files, audited recording index, and fingerprint.
- `artifacts/intel_baseline/smoke.json`: device, chosen batch size, and real-data overfit results.
- `artifacts/intel_baseline/runs/<run>/`: settings, history, resumable `latest.pt`, selected `best.pt`, predictions, and metrics.
- `artifacts/intel_baseline/validation_grid.csv`: completed configurations and validation-selected epochs.
- `artifacts/intel_baseline/selection.json`: frozen configuration, seeds, epochs, and checkpoint hashes.
- `artifacts/intel_baseline/plots/` and `test_summary.json`: generated figures and aggregate test results.
- `reports/intel_baseline.md`: current milestone status and final report.

W&B uses project `ai360-intel-baseline` and the authenticated account's default entity. Set `wandb_entity` in the experiment configuration for a team. Raw audio is never uploaded. W&B failures emit a warning and preserve local training and logs; run URLs are stored in `wandb.json` and the final report.

`report` regenerates metrics and figures from saved CSV predictions and metadata without inference or model loading. Pending runs remain explicitly marked as pending. The primary result always uses seed 42; repeat seeds do not replace it.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic data and tiny networks, including exact resumed-versus-uninterrupted training with persistent worker processes. They do not download data, upload to W&B, or claim benchmark performance. Worker tests require local IPC permissions. `smoke` separately checks the real full-size model and two normal training recordings on the GPU.
