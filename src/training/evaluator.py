"""
Evaluator: load best checkpoint and compute 9 per-site classification metrics
on the held-out test split.
"""

import json
import pathlib

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

RESULTS_DIR = pathlib.Path("results")
METRICS_FILE = RESULTS_DIR / "metrics.json"


def evaluate(
    model: nn.Module,
    test_loader,
    checkpoint_path: pathlib.Path,
    device: torch.device | None = None,
    results_path: pathlib.Path | None = None,
) -> dict[str, float]:
    """Load checkpoint, run inference on test_loader, return 9 metrics.

    Args:
        model: Any nn.Module with a compatible forward signature; weights loaded here.
        test_loader: Iterable of (x, seq_lengths, target_k_indices, labels) batches.
        checkpoint_path: Path to saved state dict (best_model.pt).
        device: Device the model lives on. Tensors are moved here before
                each forward pass. Defaults to the model's current device.
        results_path: Where to write metrics.json. Defaults to
                      ``results/metrics.json`` for backward compatibility.

    Returns:
        dict with keys: ACC, BACC, SN, SP, MCC, Kappa, F1, AUC, AUPRC.
    """
    if device is None:
        device = next(model.parameters()).device

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, seq_lengths, target_k_indices, labels in test_loader:
            x = x.to(device)
            seq_lengths = seq_lengths.to(device)
            target_k_indices = target_k_indices.to(device)
            logits = model(x, seq_lengths, target_k_indices).squeeze(1)  # (B,)
            all_logits.append(logits)
            all_labels.append(labels)

    logits_cat = torch.cat(all_logits)           # (N,)
    labels_cat = torch.cat(all_labels)            # (N,)

    y_prob = torch.sigmoid(logits_cat).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = labels_cat.cpu().numpy().astype(int)

    metrics: dict[str, float] = {
        "ACC":   float(accuracy_score(y_true, y_pred)),
        "BACC":  float(balanced_accuracy_score(y_true, y_pred)),
        "SN":    float(recall_score(y_true, y_pred)),
        "SP":    float(recall_score(y_true, y_pred, pos_label=0)),
        "MCC":   float(matthews_corrcoef(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "F1":    float(f1_score(y_true, y_pred)),
        "AUC":   float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
    }

    out = results_path if results_path is not None else METRICS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
