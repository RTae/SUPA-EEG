"""Training loops for classification and generation tasks."""

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .metrics import (
    batch_hard_triplet_loss,
    evaluate_classifier,
    evaluate_generator,
    evaluate_semantic_embeddings,
    remap_labels,
    resolve_clip_targets,
)


class BalancedBatchSampler(Sampler):
    """Yields batches with exactly `samples_per_class` samples per class.

    Each batch contains `num_classes_per_batch * samples_per_class` indices,
    guaranteeing that every anchor has at least one valid positive and one
    valid negative for batch-hard triplet mining.
    """

    def __init__(self, dataset, label_map: dict[int, int], num_classes_per_batch: int, samples_per_class: int) -> None:
        super().__init__()
        self.samples_per_class = samples_per_class
        self.num_classes_per_batch = num_classes_per_batch

        # Group dataset indices by remapped class label.
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            _, raw_label = dataset[idx]
            remapped = label_map.get(int(raw_label))
            if remapped is not None:
                groups[remapped].append(idx)
        self.groups = {k: v for k, v in groups.items() if len(v) >= samples_per_class}
        self.classes = list(self.groups.keys())
        self.num_batches = max(1, len(self.classes) // num_classes_per_batch)

    def __iter__(self):
        classes = self.classes.copy()
        random.shuffle(classes)
        for i in range(0, len(classes) - self.num_classes_per_batch + 1, self.num_classes_per_batch):
            batch_classes = classes[i : i + self.num_classes_per_batch]
            batch = []
            for cls in batch_classes:
                batch.extend(random.choices(self.groups[cls], k=self.samples_per_class))
            random.shuffle(batch)
            yield from batch

    def __len__(self) -> int:
        return self.num_batches * self.num_classes_per_batch * self.samples_per_class


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
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

        acc, acc5, test_loss = evaluate_classifier(model, eval_loader, criterion, device, label_map)
        epoch_bar.set_postfix(
            tr_loss=f"{epoch_loss / max(1, len(train_loader)):.4f}",
            eval_top1=f"{acc:.3f}",
            eval_top5=f"{acc5:.3f}",
            eval_loss=f"{test_loss:.4f}",
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


def train_semantic_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    label_map: dict[int, int],
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
            labels = remap_labels(labels, label_map)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            triplet_loss = batch_hard_triplet_loss(
                outputs["embedding"],
                labels,
                margin=triplet_margin,
            )
            triplet_loss.backward()
            optimizer.step()
            model.update_target_encoder(ema_decay)

            running_triplet += triplet_loss.item()

        top1, top5, val_loss = evaluate_semantic_embeddings(
            model,
            train_loader,
            eval_loader,
            device,
            label_map,
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
