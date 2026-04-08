"""Training loops for classification and generation tasks."""

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import evaluate_classifier, evaluate_generator, remap_labels, resolve_clip_targets


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    label_map: dict[int, int],
    save_path: str | None = None,
) -> tuple[float, int]:
    """Train a classifier, evaluate each epoch, return (best_acc, best_epoch).

    If *save_path* is given the best checkpoint is saved there.
    """
    model = model.to(device)
    best_acc, best_epoch = 0.0, -1

    epoch_bar = tqdm(range(num_epochs), desc="train", unit="ep")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        step_bar = tqdm(train_loader, desc=f"ep {epoch}", leave=False, unit="step")
        for inputs, labels in step_bar:
            labels = remap_labels(labels, label_map)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_bar.set_postfix(loss=f"{loss.item():.4f}")

        acc, test_loss = evaluate_classifier(model, test_loader, criterion, device, label_map)
        epoch_bar.set_postfix(
            tr_loss=f"{epoch_loss / max(1, len(train_loader)):.4f}",
            val_acc=f"{acc:.3f}",
            val_loss=f"{test_loss:.4f}",
        )
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    return best_acc, best_epoch


def train_generator(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    clip_embeddings: dict[str, torch.Tensor],
    save_path: str | None = None,
) -> tuple[int, float]:
    """Train the EEG-to-CLIP mapper, return (best_epoch, best_loss).

    If *save_path* is given the best checkpoint is saved there.
    """
    model = model.to(device)
    report_interval = max(len(train_loader) // 2, 1)
    best_loss, best_epoch = float("inf"), -1

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            targets = resolve_clip_targets(labels, clip_embeddings, device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % report_interval == report_interval - 1:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / report_interval:.4f}")
                running_loss = 0.0

        avg_test_loss = evaluate_generator(model, test_loader, criterion, device, clip_embeddings)
        print(f"Test loss: {avg_test_loss:.4f}")

        total_test_loss = avg_test_loss * len(test_loader)
        if total_test_loss < best_loss:
            best_loss = total_test_loss
            best_epoch = epoch
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    return best_epoch, best_loss
