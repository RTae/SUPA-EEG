import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    batch_hard_triplet_loss,
    evaluate_classifier,
    evaluate_generator,
    evaluate_semantic_embeddings,
    resolve_clip_targets,
)

def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: str | None = None,
) -> tuple[float, float, int]:
    """
    Train the classifier, return (best_top1, best_top5, best_epoch).
    
    If *save_path* is given the best checkpoint is saved there.
    """
    model = model.to(device)
    best_top1, best_top5, best_epoch = 0.0, 0.0, -1

    epoch_bar = tqdm(range(num_epochs), desc="train", unit="ep")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        top1, top5, val_loss = evaluate_classifier(model, eval_loader, criterion, device)
        epoch_bar.set_postfix(
            train_loss=f"{epoch_loss / max(1, len(train_loader)):.4f}",
            eval_top1=f"{top1:.3f}",
            eval_top5=f"{top5:.3f}",
            eval_loss=f"{val_loss:.4f}",
        )
        if top1 > best_top1:
            best_top1 = top1
            best_top5 = top5
            best_epoch = epoch
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    return best_top1, best_top5, best_epoch

def train_semantic_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    *,
    triplet_margin: float,
    ema_decay: float,
    save_path: str | None = None,
) -> tuple[float, float, int]:
    """Train semantic model with triplet loss only."""
    model = model.to(device)

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = -1

    epoch_bar = tqdm(range(num_epochs), desc="semantic-train", unit="ep")
    for epoch in epoch_bar:
        model.train()
        running_triplet = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            emb = outputs["embedding"]

            if not emb.isfinite().all():
                n_nan = (~emb.isfinite()).any(dim=1).sum().item()
                logging.warning(f"[train] epoch={epoch} — {n_nan}/{emb.shape[0]} embeddings NaN/Inf; skipping batch")
                continue

            triplet_loss = batch_hard_triplet_loss(
                emb,
                labels,
                margin=triplet_margin,
            )
            triplet_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder(ema_decay)

            running_triplet += triplet_loss.item()

        top1, top5, val_loss = evaluate_semantic_embeddings(
            model,
            eval_loader,
            device,
            triplet_margin,
        )
        epoch_bar.set_postfix(
            train_loss=f"{running_triplet / max(1, len(train_loader)):.4f}",
            eval_top1=f"{top1:.3f}",
            eval_top5=f"{top5:.3f}",
            eval_loss=f"{val_loss:.4f}",
        )

        if top1 > best_top1:
            best_top1 = top1
            best_top5 = top5
            best_epoch = epoch
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

    return best_top1, best_top5, best_epoch


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

