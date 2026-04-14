"""Inference (prediction-only) loops for classification and generation."""

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def infer_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_predictions, all_true_labels) as numpy arrays."""
    model.eval()
    all_preds, all_true = [], []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        preds = model(inputs).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_true)


@torch.no_grad()
def infer_generator(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return a list of predicted embedding batches."""
    model.eval()
    results = []
    for inputs, _labels in dataloader:
        results.append(model(inputs.to(device)).cpu())
    return results
