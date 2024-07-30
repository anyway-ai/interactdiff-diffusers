import os
import argparse
import yaml
import json
from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler

from models.unet.unet import InteractDiffusionUNet2DConditionModel
from models.text_encoder.text_encoder import FrozenCLIPEmbedder

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_constant_schedule_with_warmup
from transformers import CLIPTextModel

from dataset.concat_dataset import ConCatDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from preprocessor import preprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_one_step(batch, models):
    batch = batch_to_device(batch, device)

    x_start, timesteps, context, grounding_input = get_input(batch, models)

    noise = torch.randn_like(x_start)
    x_noisy = models['scheduler'].add_noise(original_samples = x_start, noise = noise, timesteps = timesteps)

    cross_attention_kwargs = {}
    cross_attention_kwargs['gligen'] = grounding_input
    
    output = models['unet'](x_noisy, timesteps, encoder_hidden_states = context, cross_attention_kwargs = cross_attention_kwargs)
    
    loss = torch.nn.functional.mse_loss(output.sample, noise)

    return loss


def get_input(batch, models):
    z = models['vae'].encode(batch["image"]).latent_dist.sample()
    context = models['text_encoder'].encode(batch["caption"])

    _t = torch.rand(z.shape[0]).to(z.device)
    t = (torch.pow(_t, 1) * 1000).long()
    t = torch.where(t != 1000, t, 999)

    grounding_input = preprocessor(batch)
    
    return z, t, context, grounding_input

def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch

def wrap_loader(loader):
    while True:
        for batch in loader:
            yield batch

def get_dataloader(config):
    train_dataset = ConCatDataset(
        config["train_dataset_names"],
        "/home/interactdiff-diffusers/DATA",
        train=True,
        repeats=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=True,
        num_workers=config["training_params"]["num_workers"],
        pin_memory=True,
        sampler=None,
    )

    train_loader = wrap_loader(train_loader)
    return train_loader


def create_opt(config, unet):
    params = []
    trainable_layers = []
    for name, param in unet.named_parameters():
        if ("transformer_blocks" in name) and ("fuser" in name):
            params.append(param)
            trainable_layers.append(name)
        elif "position_net" in name:
            params.append(param)
            trainable_layers.append(name)
        elif "downsample_net" in name:
            trainable_layers.append(name)

    opt = torch.optim.AdamW(
        params=params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scaler = GradScaler()
    lr_scheduler = get_constant_schedule_with_warmup(
        opt, num_warmup_steps=int(config["lr_warmup_steps"])
    )

    return opt, scaler, lr_scheduler, params, trainable_layers


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def load_custom_unet(config):
    with open(f"{config}/config.json", "r") as f:
        unet_configs = json.load(f)

    del unet_configs["_class_name"]
    del unet_configs["_diffusers_version"]

    unet = InteractDiffusionUNet2DConditionModel(**unet_configs).to(device)
    return unet


def load_models(config):
    unet = load_custom_unet(config["unet"])
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=config["vae"], variant="fp16"
    ).to(device)
    text_encoder = FrozenCLIPEmbedder().to(device)
    scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_name_or_path=config["scheduler"]
    )

    vae.eval()
    text_encoder.eval()

    disable_grads(vae)
    disable_grads(text_encoder)

    models = {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "scheduler": scheduler,
    }

    return models


def train(config):
    models = load_models(config["models_path"])
    opt, scaler, lr_scheduler, _, _ = create_opt(config['training_params'], models['unet'])
    train_loader = get_dataloader(config)

    models['unet'].train()

    for idx in tqdm(range(int(config['training_params']['num_iterations']))):
        batch = next(train_loader)
        batch_to_device(batch, device)

        loss = run_one_step(batch, models)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        lr_scheduler.step()
        opt.zero_grad()

        print(f"Loss at Iteration {idx} : {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file_path", type=str, required=False, default="./configs/train.yaml"
    )

    args = parser.parse_args()
    with open(args.config_file_path, "r") as f:
        config = yaml.safe_load(f)

    train(config)