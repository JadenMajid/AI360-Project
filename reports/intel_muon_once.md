# One-run Muon comparison

Muon improved test ROC-AUC by **+0.00472** (+0.47 percentage points) against Adam with the same seed (42).

| Metric | Adam | Muon + auxiliary Adam |
|---|---:|---:|
| Selected epoch | 47 | 42 |
| Validation ROC-AUC | 0.81552 | 0.82129 |
| Test ROC-AUC | 0.81777 | 0.82249 |
| Test macro defect ROC-AUC | 0.84967 | 0.85283 |
| Training-run wall time (minutes) | 7.30 | 7.99 |
| Peak allocated GPU memory (GiB) | 0.534 | 0.565 |

Both runs use the same audited data, FFT 1,366, bottleneck 48, width 1,024, 192-frame chunks, batch size 16, 50 epochs, peak LR 1e-4, one-cycle schedule, zero weight decay, and seed 42. Only normal training recordings are optimized. Each checkpoint is selected by validation AUC, then frozen before evaluation on the same 1,732 test recordings.

Muon updates eight hidden Conv1d/ConvTranspose1d kernels, flattened as `[dim0, remaining dimensions]`. The input/output kernels, all biases, and output PReLU use Adam. Native PyTorch Muon uses Nesterov momentum 0.95, five Newton–Schulz steps, and `match_rms_adamw` LR scaling. W&B is disabled; logs and checkpoints are local.

![Training and validation curves](../artifacts/intel_muon_once/comparison.png)

| Defect (test) | Adam AUC | Muon AUC | Difference |
|---|---:|---:|---:|
| Burnthrough | 0.71152 | 0.66963 | -0.04189 |
| Excessive Convexity | 0.95072 | 0.94690 | -0.00382 |
| Undercut | 0.98388 | 0.98140 | -0.00248 |
| Overlap | 0.99928 | 0.99969 | +0.00041 |
| Crater Cracks | 0.98816 | 0.98929 | +0.00112 |
| Spatter | 0.75837 | 0.76074 | +0.00238 |
| Excessive Penetration | 0.62610 | 0.62913 | +0.00303 |
| Warping | 0.61617 | 0.62970 | +0.01353 |
| Porosity w Excessive Penetration | 0.94060 | 0.95579 | +0.01519 |
| Porosity | 0.94424 | 0.96509 | +0.02086 |
| Lack of Fusion | 0.82738 | 0.85377 | +0.02639 |

This is one untuned optimizer trial. It does not establish a multi-seed result or generalization confidence interval. Recorded wall time includes validation and checkpoint saves, and excludes feature preparation, smoke testing, and final test inference; it is not a controlled speed benchmark.

Validation and test AUCs were recomputed from saved predictions. Checkpoint hashes, test provenance, recording identities, labels, and dataset fingerprints were verified. All 16 regression tests passed, including exact Muon checkpoint resumption; the full-size real-data GPU smoke check also passed.

Reproduce or resume the same run:

```bash
bash scripts/run_intel_muon_once.sh
```

Configuration: `configs/intel_muon_once.json`. Artifacts: `artifacts/intel_muon_once/`, including `comparison.json`, `comparison.png`, `run.log`, selected checkpoints, and source snapshots.

Implementation references: [PyTorch Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html), [Muon author repository](https://github.com/KellerJordan/Muon).
