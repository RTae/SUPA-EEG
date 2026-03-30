import os

import torch
from PIL import Image
from tqdm import tqdm
import typer
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from dataset import EEGImageNetDataset
from utilities import (
    Args, BatchSize, DatasetDir, Granularity, Model,
    OutputDir, PretrainedModel, Subject, get_device,
)

SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"

BLIP_GENERATION_CONFIG = {
    "max_length": 200,
    "num_beams": 20,
    "temperature": 0.5,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 2.0,
    "do_sample": True,
}


def generate_clip_embeddings(dataset: EEGImageNetDataset, output_dir: str, device: torch.device) -> None:
    """Caption every image with BLIP, then encode captions into CLIP text embeddings."""
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, local_files_only=True)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_NAME, use_safetensors=True, local_files_only=True,
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        SD_MODEL_NAME, subfolder="tokenizer", local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        SD_MODEL_NAME, subfolder="text_encoder",
        use_safetensors=True, local_files_only=True,
    ).to(device)

    embeddings: dict[str, torch.Tensor] = {}
    caption_path = os.path.join(output_dir, "caption.txt")

    for image_name in tqdm(dataset.images, desc="Generating CLIP embeddings"):
        image_path = os.path.join(dataset.dataset_dir, "imageNet_images", image_name.split("_")[0], image_name)
        raw_image = Image.open(image_path).convert("RGB")

        blip_inputs = processor(images=raw_image, return_tensors="pt").to(device)
        out = blip_model.generate(**blip_inputs, **BLIP_GENERATION_CONFIG)
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


def main(
    dataset_dir: DatasetDir = "data/",
    granularity: Granularity = "all",
    model: Model = "mlp_sd",
    batch_size: BatchSize = 40,
    subject: Subject = 0,
    output_dir: OutputDir = "output/",
    pretrained_model: PretrainedModel = None,
) -> None:
    args = Args(dataset_dir, granularity, model, batch_size, subject, output_dir, pretrained_model)
    print(args)

    device = get_device()
    dataset = EEGImageNetDataset.from_args(args, map_location=device)
    generate_clip_embeddings(dataset, args.output_dir, device)


if __name__ == "__main__":
    typer.run(main)
