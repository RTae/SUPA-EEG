import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from dataset import EEGImageNetDataset
from utilities import get_device


def generate_clip_embeddings(
    dataset: EEGImageNetDataset, output_dir: str, blip_cfg, sd_model: str, device: torch.device,
) -> None:
    """Caption every image with BLIP, then encode captions into CLIP text embeddings."""
    processor = BlipProcessor.from_pretrained(blip_cfg.model, local_files_only=True)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_cfg.model, use_safetensors=True, local_files_only=True,
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        sd_model, subfolder="tokenizer", local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        sd_model, subfolder="text_encoder",
        use_safetensors=True, local_files_only=True,
    ).to(device)

    generation_config = OmegaConf.to_container(blip_cfg.generation, resolve=True)

    embeddings: dict[str, torch.Tensor] = {}
    caption_path = os.path.join(output_dir, "caption.txt")

    for image_name in tqdm(dataset.images, desc="Generating CLIP embeddings"):
        image_path = os.path.join(dataset.dataset_dir, "imageNet_images", image_name.split("_")[0], image_name)
        raw_image = Image.open(image_path).convert("RGB")

        blip_inputs = processor(images=raw_image, return_tensors="pt").to(device)
        out = blip_model.generate(**blip_inputs, **generation_config)
        caption = processor.decode(out[0], skip_special_tokens=True)

        with open(caption_path, "a", encoding="utf-8") as f:
            f.write(f"{image_name}\t{caption}\n")

        clip_inputs = tokenizer(
            caption, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(clip_inputs.input_ids)[0]
        embeddings[image_name] = text_embeddings.cpu()

    torch.save(embeddings, os.path.join(output_dir, "clip_embeddings.pth"))


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    dataset = EEGImageNetDataset.from_args(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    generate_clip_embeddings(dataset, cfg.output_dir, cfg.blip, cfg.diffusion.sd_model, device)


if __name__ == "__main__":
    main()
