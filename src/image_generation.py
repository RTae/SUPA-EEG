import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import EEGImageNetDataset
from de_feat_cal import de_feat_cal
from model.mlp_sd import MLPMapper
from utilities import build_arg_parser, get_device


def model_init(model_name: str) -> torch.nn.Module:
    if model_name.lower() == "mlp_sd":
        return MLPMapper()
    raise ValueError(f"Unknown model: {model_name}")


def train(
    args,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    clip_embeddings: dict[str, torch.Tensor],
) -> tuple[int, float]:
    model = model.to(device)
    report_interval = max(len(train_loader) // 2, 1)
    best_loss = float("inf")
    best_epoch = -1

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # -- train --
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            targets = torch.stack([clip_embeddings[name] for name in labels]).squeeze()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % report_interval == report_interval - 1:
                print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / report_interval:.4f}")
                running_loss = 0.0

        # -- evaluate --
        model.eval()
        with torch.no_grad():
            test_loss = sum(
                criterion(
                    model(inputs.to(device)),
                    torch.stack([clip_embeddings[n] for n in labels]).squeeze().to(device),
                ).item()
                for inputs, labels in test_loader
            )
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test loss: {avg_test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"mlpsd_s{args.subject}_0.pth"))

    return best_epoch, best_loss


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset.from_args(args)
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, args.subject, args.granularity)
    dataset.add_frequency_feat(de_feat)

    train_idx = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_idx = np.array([i for i in range(len(dataset)) if i % 50 >= 30])
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    device = get_device()
    model = model_init(args.model)
    clip_embeddings = torch.load(os.path.join(args.output_dir, "clip_embeddings.pth"), map_location="cpu")

    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, args.pretrained_model), map_location="cpu"))

    if args.model.lower() == "mlp_sd":
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        epoch, loss = train(args, model, train_loader, test_loader, criterion, optimizer, 1000, device, clip_embeddings)

    with open(os.path.join(args.output_dir, "mlpsd.txt"), "a", encoding="utf-8") as f:
        f.write(f"{epoch}: {loss}\n")
