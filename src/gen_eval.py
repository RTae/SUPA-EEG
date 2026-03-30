import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel

from dataset import EEGImageNetDataset
from preprocessing.de_feat_cal import de_feat_cal
from model.mlp_sd import MLPMapper
from utilities import get_device


def load_diffusion_pipeline(diff_cfg, device):
    """Load and return all Stable Diffusion components."""
    sd_model = diff_cfg.sd_model
    tokenizer = CLIPTokenizer.from_pretrained(
        sd_model, subfolder="tokenizer", local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        sd_model, subfolder="text_encoder",
        use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        sd_model, subfolder="vae",
        use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        sd_model, subfolder="unet",
        use_safetensors=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    scheduler = PNDMScheduler.from_pretrained(
        sd_model, subfolder="scheduler", local_files_only=True,
    )
    return tokenizer, text_encoder, vae, unet, scheduler


def diffusion(embeddings, tokenizer, text_encoder, vae, unet, scheduler, diff_cfg, device):
    """Run the reverse diffusion process and return generated images."""
    batch_size = embeddings.size(0)
    generator = torch.Generator(device=device).manual_seed(diff_cfg.seed)

    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length",
        max_length=tokenizer.model_max_length, return_tensors="pt",
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, embeddings])
    latents = torch.randn(
        (batch_size, unet.config.in_channels, diff_cfg.image_size // 8, diff_cfg.image_size // 8),
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(diff_cfg.num_inference_steps)

    for t in tqdm(scheduler.timesteps, desc="Diffusion", leave=False):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + diff_cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
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


def save_generated_images(run_dir, cfg, dataloader, model, clip_embeddings, device, pipeline):
    """Generate and save images for every batch in the dataloader."""
    tokenizer, text_encoder, vae, unet, scheduler = pipeline
    output_subdir = os.path.join(run_dir, f"generated_s{cfg.subject}")
    os.makedirs(output_subdir, exist_ok=True)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Generating")):
            targets = torch.stack([clip_embeddings[name] for name in labels]).squeeze()
            targets = targets.to(device=device, dtype=torch.bfloat16)
            inputs = inputs.to(device=device)
            embeddings = model(inputs).to(dtype=torch.bfloat16)
            generated = diffusion(
                embeddings, tokenizer, text_encoder, vae, unet, scheduler,
                cfg.diffusion, device,
            )
            for i, img_tensor in enumerate(generated):
                img = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
                img.save(os.path.join(output_subdir, f"{i + 1 + batch_idx * cfg.batch_size}.png"))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

    dataset = EEGImageNetDataset.from_args(cfg)
    eeg_data = np.stack([sample[0].numpy() for sample in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, cfg.subject, cfg.granularity)
    dataset.add_frequency_feat(de_feat)

    nn_model = model_init(cfg.model.name)
    clip_embeddings = torch.load(os.path.join(cfg.output_dir, "clip_embeddings.pth"), map_location="cpu")

    if cfg.pretrained_model:
        nn_model.load_state_dict(
            torch.load(os.path.join(cfg.output_dir, cfg.pretrained_model), map_location="cpu")
        )

    if cfg.model.name.lower() == "mlp_sd":
        dataset.use_frequency_feat = True
        dataset.use_image_label = True
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
        pipeline = load_diffusion_pipeline(cfg.diffusion, device)
        run_dir = HydraConfig.get().runtime.output_dir
        save_generated_images(run_dir, cfg, dataloader, nn_model, clip_embeddings, device, pipeline)


if __name__ == "__main__":
    main()
