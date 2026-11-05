# Intel audio baseline replication

16 kHz adaptation; published 192 kHz reference AUC: **0.8460**.

## Status

Dataset audit has not completed. No experimental results are claimed.

Model selection is pending. Test evaluation remains locked.

## Figures

## Checkpoints and W&B

Local artifacts: `/home/jaden/AI360-Project/artifacts/intel_baseline`.


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
