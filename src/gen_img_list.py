import os

import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import EEGImageNetDataset
from utilities import get_device, wnid2category


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    dataset = EEGImageNetDataset.from_args(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Write per-sample image filenames
    with open(os.path.join(cfg.output_dir, f"s{cfg.subject}.txt"), "w", encoding="utf-8") as f:
        dataset.use_image_label = True
        for data in dataset:
            f.write(f"{data[1]}\n")

    # Write per-class label summary (one line per 50-sample block)
    with open(os.path.join(cfg.output_dir, f"s{cfg.subject}_label.txt"), "w", encoding="utf-8") as f:
        dataset.use_image_label = False
        for idx, data in enumerate(dataset):
            if idx % 50 == 0:
                label_wnid = dataset.labels[data[1]]
                f.write(f"{idx + 1}-{idx + 50}: {wnid2category(label_wnid, 'ch')}\n")


if __name__ == "__main__":
    main()
