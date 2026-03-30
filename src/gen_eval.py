import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel

from dataset import EEGImageNetDataset
from de_feat_cal import de_feat_cal
from model.mlp_sd import MLPMapper
from utilities import build_arg_parser, get_device

SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5


def load_diffusion_pipeline(device: torch.device):
    """Load and return all Stable Diffusion components."""
    tokenizer = CLIPTokenizer.from_pretrained(
        SD_MODEL_NAME, subfolder="tokenizer", local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        SD_MODEL_NAME, subfolder="text_encoder",
        use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        SD_MODEL_NAME, subfolder="vae",
        use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        SD_MODEL_NAME, subfolder="unet",
        use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    scheduler = PNDMScheduler.from_pretrained(
        SD_MODEL_NAME, subfolder="scheduler", local_files_only=True,
    )
    return tokenizer, text_encoder, vae, unet, scheduler


def diffusion(embeddings, tokenizer, text_encoder, vae, unet, scheduler, device):
    """Run the reverse diffusion process and return generated images."""
    batch_size = embeddings.size(0)
    generator = torch.Generator(device=device).manual_seed(42)

    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length",
        max_length=tokenizer.model_max_length, return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, embeddings])
    latents = torch.randn(
        (batch_size, unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    for t in tqdm(scheduler.timesteps, desc="Diffusion", leave=False):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    with torch.no_grad():
        decoded = vae.decode(latents).sample
    images = ((decoded / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    return images


def model_init(model_name: str) -> torch.nn.Module:
    if model_name.lower() == "mlp_sd":
        return MLPMapper()
    raise ValueError(f"Unknown model: {model_name}")


def save_generated_images(args, dataloader, model, clip_embeddings, device, pipeline):
    """Generate and save images for every batch in the dataloader."""
    tokenizer, text_encoder, vae, unet, scheduler = pipeline
    output_subdir = os.path.join(args.output_dir, f"generated_s{args.subject}")
    os.makedirs(output_subdir, exist_ok=True)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Generating")):
            targets = torch.stack([clip_embeddings[name] for name in labels]).squeeze()
            targets = targets.to(device=device, dtype=torch.bfloat16)
            inputs = inputs.to(device=device)
            embeddings = model(inputs).to(dtype=torch.bfloat16)
            generated = diffusion(embeddings, tokenizer, text_encoder, vae, unet, scheduler, device)
            for i, img_tensor in enumerate(generated):
                img = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
                img.save(os.path.join(output_subdir, f"{i + 1 + batch_idx * args.batch_size}.png"))


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset.from_args(args)
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, args.subject, args.granularity)
    dataset.add_frequency_feat(de_feat)

    device = get_device()
    model = model_init(args.model)
    clip_embeddings = torch.load(os.path.join(args.output_dir, "clip_embeddings.pth"), map_location="cpu")

    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, args.pretrained_model), map_location="cpu"))

    if args.model.lower() == "mlp_sd":
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        pipeline = load_diffusion_pipeline(device)
        save_generated_images(args, dataloader, model, clip_embeddings, device, pipeline)
