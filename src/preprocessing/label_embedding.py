import json
import os

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import EEGImageNetDataset
from utilities import get_device, wnid2category


def _label_text(label_name: str, dataset_dir: str) -> str:
    try:
        category = wnid2category(label_name, "en", os.path.join(dataset_dir, "imageNet_images"))
    except Exception:
        category = label_name
    category = category.replace("_", " ")
    return f"a photo of {category}"


def _encode_label_texts(texts: list[str], sd_model: str, device: torch.device) -> torch.Tensor:
    tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        sd_model,
        subfolder="text_encoder",
        use_safetensors=True,
        local_files_only=True,
    ).to(device)
    text_encoder.eval()

    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        hidden = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, dim=1)
    return pooled.cpu()


def build_semantic_knn(
    dataset: EEGImageNetDataset,
    output_path: str,
    sd_model: str,
    k_pos: int,
    k_neg: int,
    device: torch.device,
    overwrite: bool = False,
) -> None:
    if os.path.exists(output_path) and not overwrite:
        print(f"Semantic KNN cache already exists, skip: {output_path}")
        return

    label_names = list(dataset.labels)
    label_texts = [_label_text(name, dataset.dataset_dir) for name in label_names]
    embeddings = _encode_label_texts(label_texts, sd_model, device)

    sim = embeddings @ embeddings.t()
    n = sim.shape[0]
    records: dict[str, dict] = {}

    for idx in range(n):
        row = sim[idx].clone()
        row[idx] = -1.0
        pos_ids = torch.topk(row, k=min(k_pos, n - 1), largest=True).indices.tolist()

        far = sim[idx].clone()
        far[idx] = 1.0
        neg_ids = torch.topk(far, k=min(k_neg, n - 1), largest=False).indices.tolist()

        records[str(idx)] = {
            "label_name": label_names[idx],
            "text_prompt": label_texts[idx],
            "positives": pos_ids,
            "negatives": neg_ids,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_labels": n,
                "k_pos": k_pos,
                "k_neg": k_neg,
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(f"Saved semantic KNN file to: {output_path}")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = get_device()
    dataset = EEGImageNetDataset(
        dataset_dir=cfg.dataset_dir,
        subject=-1,
        granularity=cfg.granularity,
        map_location=device,
    )

    default_path = os.path.join(cfg.dataset_dir, f"semantic_knn_{cfg.granularity}.json")
    output_path = cfg.model.get("semantic_knn_path", default_path)
    k_pos = int(cfg.model.get("semantic_knn_k", 4))
    k_neg = int(cfg.model.get("semantic_neg_k", 4))

    build_semantic_knn(
        dataset=dataset,
        output_path=output_path,
        sd_model=cfg.diffusion.sd_model,
        k_pos=k_pos,
        k_neg=k_neg,
        device=device,
        overwrite=bool(cfg.model.get("semantic_knn_overwrite", False)),
    )


if __name__ == "__main__":
    main()
