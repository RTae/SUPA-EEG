import os

import typer

from dataset import EEGImageNetDataset
from utilities import (
    Args, BatchSize, DatasetDir, Granularity, Model,
    OutputDir, PretrainedModel, Subject, get_device, wnid2category,
)


def main(
    dataset_dir: DatasetDir = "data/",
    granularity: Granularity = "coarse",
    model: Model = "eegnet",
    batch_size: BatchSize = 40,
    subject: Subject = 0,
    output_dir: OutputDir = "output/",
    pretrained_model: PretrainedModel = None,
) -> None:
    args = Args(dataset_dir, granularity, model, batch_size, subject, output_dir, pretrained_model)
    print(args)

    device = get_device()
    dataset = EEGImageNetDataset.from_args(args, map_location=device)

    # Write per-sample image filenames
    with open(os.path.join(args.output_dir, f"s{args.subject}.txt"), "w", encoding="utf-8") as f:
        dataset.use_image_label = True
        for data in dataset:
            f.write(f"{data[1]}\n")

    # Write per-class label summary (one line per 50-sample block)
    with open(os.path.join(args.output_dir, f"s{args.subject}_label.txt"), "w", encoding="utf-8") as f:
        dataset.use_image_label = False
        for idx, data in enumerate(dataset):
            if idx % 50 == 0:
                label_wnid = dataset.labels[data[1]]
                f.write(f"{idx + 1}-{idx + 50}: {wnid2category(label_wnid, 'ch')}\n")


if __name__ == "__main__":
    typer.run(main)
