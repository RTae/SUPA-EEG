import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from dataset import EEGImageNetDataset
from preprocessing.de_feat_cal import de_feat_cal
from model.mlp_sd import MLPMapper
from trainer import train_generator
from utilities import get_device, build_optimizer


def model_init(model_name: str) -> torch.nn.Module:
    if model_name.lower() == "mlp_sd":
        return MLPMapper()
    raise ValueError(f"Unknown model: {model_name}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

    dataset = EEGImageNetDataset.from_args(cfg)
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, cfg.subject, cfg.granularity)
    dataset.add_frequency_feat(de_feat)

    train_idx = np.array([i for i in range(len(dataset)) if i % 50 < 30])
    test_idx = np.array([i for i in range(len(dataset)) if i % 50 >= 30])
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    nn_model = model_init(cfg.model.name)
    clip_embeddings = torch.load(os.path.join(cfg.output_dir, "clip_embeddings.pth"), map_location="cpu")

    if cfg.pretrained_model:
        nn_model.load_state_dict(
            torch.load(os.path.join(cfg.output_dir, cfg.pretrained_model), map_location="cpu")
        )

    if cfg.model.name.lower() == "mlp_sd":
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)
        optimizer = build_optimizer(nn_model.parameters(), cfg.model.optimizer)
        criterion = torch.nn.MSELoss()
        run_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(run_dir, f"mlpsd_s{cfg.subject}_0.pth")
        epoch, loss = train_generator(
            nn_model, train_loader, test_loader, criterion, optimizer,
            cfg.model.epochs, device, clip_embeddings, save_path=save_path,
        )

    with open(os.path.join(run_dir, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"{epoch}: {loss}\n")


if __name__ == "__main__":
    main()
