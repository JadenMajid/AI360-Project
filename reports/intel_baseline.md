# Intel audio baseline replication

16 kHz adaptation; published 192 kHz reference AUC: **0.8460**.

## Status

Configured FFT sizes: [1366]; bottlenecks: [48]; seeds: [42, 43, 44]; 50 epochs per run.

Dataset audit passed for 4,040 recordings.
Manifest SHA-256: `9c4fb4df70bb137fc851dd858ac22d53eed86ba08c3806d7177e40b300406826`.
Data fingerprint: `d250a3293b81d849601869fc56cb36a96a7d665a0c8007ecba2d3da3aeb0102d`.

Validation grid: 1/1 configurations completed.

| FFT | Bottleneck | Best validation AUC | Epoch |
|---:|---:|---:|---:|
| 1366 | 48 | 0.81552 | 47 |

Frozen primary run: `fft1366-bn48-seed42`.

| Run | Seed | Test AUC | Macro defect AUC |
|---|---:|---:|---:|
| fft1366-bn48-seed42 | 42 | 0.81777 | 0.84967 |
| fft1366-bn48-seed43 | 43 | 0.81745 | 0.84926 |
| fft1366-bn48-seed44 | 44 | 0.81880 | 0.85054 |

Test AUC across 3 seeds: **0.81801 ± 0.00070** (sample standard deviation).

Primary test AUC: **0.81777**; difference from published reference: **-0.02823**.

| Defect | Replicated AUC | Published 192 kHz AUC |
|---|---:|---:|
| Burnthrough | 0.71152 | 0.7018 |
| Crater_Cracks | 0.98816 | 0.9453 |
| Excessive_Convexity | 0.95072 | 0.8842 |
| Excessive_Penetration | 0.62610 | 0.7807 |
| Lack_of_Fusion | 0.82738 | 0.8372 |
| Overlap | 0.99928 | 0.9743 |
| Porosity | 0.94424 | 0.9949 |
| Porosity_w_Excessive_Penetration | 0.94060 | 0.9659 |
| Spatter | 0.75837 | 0.7796 |
| Undercut | 0.98388 | 0.8989 |
| Warping | 0.61617 | 0.7648 |

## Figures

![per defect](../artifacts/intel_baseline/plots/per_defect.png)

![test roc](../artifacts/intel_baseline/plots/test_roc.png)

![training curves](../artifacts/intel_baseline/plots/training_curves.png)

![validation grid](../artifacts/intel_baseline/plots/validation_grid.png)

## Checkpoints and W&B

Local artifacts: `/home/jaden/AI360-Project/artifacts/intel_baseline`.

- `/home/jaden/AI360-Project/artifacts/intel_baseline/runs/fft1366-bn48-seed42/best.pt` — [W&B run](https://forge.coreweave.com/wandb/majid-jaden-university-of-british-columbia/ai360-intel-baseline/runs/qgnghkbj)
- `/home/jaden/AI360-Project/artifacts/intel_baseline/runs/fft1366-bn48-seed43/best.pt` — [W&B run](https://forge.coreweave.com/wandb/majid-jaden-university-of-british-columbia/ai360-intel-baseline/runs/8acxqwjx)
- `/home/jaden/AI360-Project/artifacts/intel_baseline/runs/fft1366-bn48-seed44/best.pt` — [W&B run](https://forge.coreweave.com/wandb/majid-jaden-university-of-british-columbia/ai360-intel-baseline/runs/49ib9wjm)

## Reproduction

Run from the repository root after installing requirements and authenticating W&B:

```bash
.venv/bin/python -m src.intel_baseline prepare
.venv/bin/python -m src.intel_baseline smoke
.venv/bin/python -m src.intel_baseline sweep --resume
.venv/bin/python -m src.intel_baseline evaluate
.venv/bin/python -m src.intel_baseline report
```

Use the same `--config`, `--data-root`, and `--output-dir` for all commands when overriding defaults.

## Reproduction assumptions

- Public mono 16 kHz audio replaces the paper's 192 kHz input; FFT lengths preserve duration, rounded to even integers.
- Full recordings, channel zero, no trimming or augmentation; Hann STFT, half-window hop, no centering, natural log magnitude floored at 1e-8.
- BatchNorm has no affine parameters; standardized features are the detached reconstruction target. Only training updates statistics.
- Five unpadded encoder convolutions and five transposed decoder convolutions; hidden width 1,024, LeakyReLU slope 0.01, scalar output PReLU initialized to 0.25.
- 192-frame chunks include an end-aligned final chunk. Short inputs are reflect-padded; padding is excluded from loss and scores, but enters convolution and training BatchNorm.
- Audio shorter than one FFT window is reflect-padded to one window. An incomplete trailing STFT window is ignored.
- Overlapping errors are averaged per original frame before the recording mean; higher scores indicate anomalies.
- Adam, 50 epochs, peak LR 1e-4; cosine one-cycle warmup 0.3, division factors 25/10,000, fixed Adam momentum, no weight decay.
- Validation selects checkpoint and configuration; test labels never influence selection or seed choice.
- These choices fill omissions in the welding paper. Actual parameter counts are recorded in each run's settings.json.

Source: [Stemmer et al., 2024](https://arxiv.org/abs/2409.02290), Tables 4, 5, 7, and 9.
