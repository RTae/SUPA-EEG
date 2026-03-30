import os

from dataset import EEGImageNetDataset
from utilities import build_arg_parser, get_device, wnid2category


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)
    
    # Device
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
