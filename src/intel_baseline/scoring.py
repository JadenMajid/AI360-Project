import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def metrics(predictions):
    if predictions.recording_id.duplicated().any() or not np.isfinite(predictions.score).all():
        raise ValueError("Expected one finite score per recording")
    if set(predictions.label.unique()) != {0, 1}:
        raise ValueError("ROC-AUC requires both normal and defective recordings")
    per_defect = {}
    for category in sorted(set(predictions.category) - {"Good"}):
        subset = predictions[predictions.category.isin(["Good", category])]
        per_defect[category] = float(roc_auc_score(subset.label, subset.score))
    return dict(auc=float(roc_auc_score(predictions.label, predictions.score)), per_defect=per_defect,
                macro_defect_auc=float(np.mean(list(per_defect.values()))))


@torch.inference_mode()
def score_recordings(model, dataset, loader, device):
    model.eval()
    sums = [np.zeros(n, dtype=np.float64) for n in dataset.n_frames]
    counts = [np.zeros(n, dtype=np.int64) for n in dataset.n_frames]
    for features, lengths, recording_ids, starts in loader:
        reconstructed, target = model(features.to(device))
        errors = (reconstructed - target).square().mean(dim=1).cpu().numpy()
        if not np.isfinite(errors).all():
            raise RuntimeError("Non-finite reconstruction errors")
        for error, length, recording, start in zip(errors, lengths.tolist(), recording_ids.tolist(), starts.tolist()):
            sums[recording][start:start + length] += error[:length]
            counts[recording][start:start + length] += 1
    if any((count == 0).any() for count in counts):
        raise ValueError("Evaluation did not cover every original frame")
    result = dataset.frame[["recording_id", "split", "category", "label"]].copy()
    result["score"] = [float((total / count).mean()) for total, count in zip(sums, counts)]
    result["n_frames"] = dataset.n_frames
    return result
