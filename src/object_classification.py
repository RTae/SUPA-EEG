import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import EEGImageNetDataset
from de_feat_cal import de_feat_cal
from model.eegnet import EEGNet
from model.mlp import MLP
from model.rgnn import RGNN, get_edge_weight
from model.simple_model import SimpleModel
from utilities import build_arg_parser, get_device

SIMPLE_MODELS = {"svm", "rf", "knn", "dt", "ridge"}

# Per-model training hyperparameters
MODEL_CONFIGS: dict[str, dict] = {
    "eegnet": {
        "use_freq": False,
        "criterion": torch.nn.CrossEntropyLoss,
        "optimizer": lambda p: optim.SGD(p, lr=1e-2, weight_decay=1e-3, momentum=0.9),
    },
    "mlp": {
        "use_freq": True,
        "criterion": torch.nn.CrossEntropyLoss,
        "optimizer": lambda p: optim.SGD(p, lr=1e-4, weight_decay=1e-4, momentum=0.9),
    },
    "rgnn": {
        "use_freq": True,
        "criterion": torch.nn.CrossEntropyLoss,
        "optimizer": lambda p: optim.Adam(p, lr=1e-3),
    },
}


def model_init(args, is_simple: bool, num_classes: int, device: torch.device) -> object:
    name = args.model.lower()
    if is_simple:
        return SimpleModel(args)
    if name == "eegnet":
        return EEGNet(args, num_classes)
    if name == "mlp":
        return MLP(args, num_classes)
    if name == "rgnn":
        edge_index, edge_weight = get_edge_weight()
        return RGNN(device, 62, edge_weight, edge_index, 5, 200, num_classes, 2)
    raise ValueError(f"Unknown model: {args.model}")


def train_and_evaluate(
    args,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    all_labels: np.ndarray,
) -> tuple[float, int]:
    model = model.to(device)
    unique_labels = torch.from_numpy(all_labels).unique()
    label_map = {orig.item(): new for new, orig in enumerate(unique_labels)}

    report_interval = max(len(train_loader) // 2, 1)
    best_acc = 0.0
    best_epoch = -1

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # -- train --
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            labels = torch.tensor([label_map[l.item()] for l in labels])
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % report_interval == report_interval - 1:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / report_interval:.4f}")
                running_loss = 0.0

        # -- evaluate --
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = torch.tensor([label_map[l.item()] for l in labels])
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                predicted = outputs.argmax(dim=1)
                total += len(labels)
                correct += accuracy_score(labels.cpu(), predicted.cpu(), normalize=False)

        acc = correct / total
        print(f"Accuracy: {acc:.4f}  |  Test loss: {test_loss / len(test_loader):.4f}")
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"eegnet_s{args.subject}_1x_22.pth"))

    return best_acc, best_epoch


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset.from_args(args)
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, args.subject, args.granularity)
    dataset.add_frequency_feat(de_feat)

    all_labels = np.array([sample[1] for sample in dataset])
    train_idx = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_idx = np.array([i for i in range(len(dataset)) if i % 50 >= 30])
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    model_name = args.model.lower()
    is_simple = model_name in SIMPLE_MODELS
    device = get_device()
    model = model_init(args, is_simple, len(dataset) // 50, device)

    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, args.pretrained_model), map_location="cpu"))

    if is_simple:
        model.fit(de_feat[train_idx], all_labels[train_idx])
        acc = accuracy_score(all_labels[test_idx], model.predict(de_feat[test_idx]))
        with open(os.path.join(args.output_dir, "simple.txt"), "a", encoding="utf-8") as f:
            f.write(f"{acc}\n")
    else:
        cfg = MODEL_CONFIGS[model_name]
        dataset.use_frequency_feat = cfg["use_freq"]
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        criterion = cfg["criterion"]()
        optimizer = cfg["optimizer"](model.parameters())
        acc, epoch = train_and_evaluate(
            args, model, train_loader, test_loader, criterion, optimizer, 1000, device, all_labels,
        )
        with open(os.path.join(args.output_dir, "eegnet.txt"), "a", encoding="utf-8") as f:
            f.write(f"{epoch}: {acc}\n")
