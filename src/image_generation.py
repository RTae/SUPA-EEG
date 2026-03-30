import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import EEGImageNetDataset
from de_feat_cal import de_feat_cal
from model.mlp_sd import MLPMapper
from trainer import train_generator
from utilities import build_arg_parser, get_device


def model_init(model_name: str) -> torch.nn.Module:
    if model_name.lower() == "mlp_sd":
        return MLPMapper()
    raise ValueError(f"Unknown model: {model_name}")


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
        save_path = os.path.join(args.output_dir, f"mlpsd_s{args.subject}_0.pth")
        epoch, loss = train_generator(
            model, train_loader, test_loader, criterion, optimizer, 1000, device, clip_embeddings,
            save_path=save_path,
        )

    with open(os.path.join(args.output_dir, "mlpsd.txt"), "a", encoding="utf-8") as f:
        f.write(f"{epoch}: {loss}\n")
