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

from dataset.concat_dataset import ConCatDataset
from torch.utils.data import DataLoader

from preprocessor import preprocessor

device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer():
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models = self.load_models(self.config["models_path"])
        opt, scaler, lr_scheduler, _, _ = self.create_opt(self.config['training_params'], self.models['unet'])
        train_loader = self.get_dataloader()

        self.models['unet'].train()
        
        self.train()

    def run_one_step(self, batch):
        batch = self.batch_to_device(batch, device)

        x_start, timesteps, context, grounding_input = self.get_input(batch, self.models)

        noise = torch.randn_like(x_start)
        x_noisy = self.models['scheduler'].add_noise(original_samples = x_start, noise = noise, timesteps = timesteps)

        cross_attention_kwargs = {}
        cross_attention_kwargs['gligen'] = grounding_input
        
        output = self.models['unet'](x_noisy, timesteps, encoder_hidden_states = context, cross_attention_kwargs = cross_attention_kwargs)
        
        loss = torch.nn.functional.mse_loss(output.sample, noise)

        return loss


    def get_input(self, batch):
        z = self.models['vae'].encode(batch["image"]).latent_dist.sample()
        context = self.models['text_encoder'].encode(batch["caption"])

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t != 1000, t, 999)

        grounding_input = preprocessor(batch)
        
        return z, t, context, grounding_input

    def batch_to_device(self, batch, device):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch

    def wrap_loader(self, loader):
        while True:
            for batch in loader:
                yield batch

    def get_dataloader(self):
        train_dataset = ConCatDataset(
            self.config["train_dataset_names"],
            "/home/interactdiff-diffusers/DATA",
            train=True,
            repeats=None,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training_params"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training_params"]["num_workers"],
            pin_memory=True,
            sampler=None,
        )

        train_loader = self.wrap_loader(train_loader)
        return train_loader


    def create_opt(self, config):
        params = []
        trainable_layers = []
        for name, param in self.models['unet'].named_parameters():
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


    def disable_grads(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def load_custom_unet(self, config):
        with open(f"{config}/config.json", "r") as f:
            unet_configs = json.load(f)

        del unet_configs["_class_name"]
        del unet_configs["_diffusers_version"]

        unet = InteractDiffusionUNet2DConditionModel(**unet_configs).to(device)
        return unet


    def load_models(self, config):
        unet = self.load_custom_unet(config["unet"])
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=config["vae"], variant="fp16"
        ).to(device)
        text_encoder = FrozenCLIPEmbedder().to(device)
        scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path=config["scheduler"]
        )

        vae.eval()
        text_encoder.eval()

        self.disable_grads(vae)
        self.disable_grads(text_encoder)

        models = {
            "unet": unet,
            "vae": vae,
            "text_encoder": text_encoder,
            "scheduler": scheduler,
        }

        return models


    def train(self):
        for idx in tqdm(range(int(self.config['training_params']['num_iterations']))):
            batch = next(self.train_loader)
            self.batch_to_device(batch, device)

            loss = self.run_one_step(batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.lr_scheduler.step()
            self.opt.zero_grad()

            print(f"Loss at Iteration {idx} : {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file_path", type=str, required=False, default="./configs/train.yaml"
    )

    args = parser.parse_args()