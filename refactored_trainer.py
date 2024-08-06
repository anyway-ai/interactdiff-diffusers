import os
import argparse
import yaml
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_constant_schedule_with_warmup
from diffusers.models.attention import GatedSelfAttentionDense
from diffusers.utils.torch_utils import randn_tensor


from transformers import CLIPTokenizer, CLIPTextModel

from safetensors.torch import load_file

from models.unet.unet import InteractDiffusionUNet2DConditionModel
from dataset.concat_dataset import ConCatDataset
from preprocessor import prepare_interactdiff_inputs

device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer():
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.l_simple_weight = 1

        self.models = self.load_models(self.config["models_path"])
        self.opt, self.scaler, self.lr_scheduler, _, _ = self.create_opt(self.config['training_params'])
        self.train_loader = self.get_dataloader()

        self.models['unet'].train()
        self.vae_scale_factor = 2 ** (len(self.models['vae'].config.block_out_channels) - 1)
        self.height = self.models['unet'].config.sample_size * self.vae_scale_factor
        self.width = self.models['unet'].config.sample_size * self.vae_scale_factor
    
    def encode_caption(self, caption):
        text_inputs = self.models['tokenizer'](
                caption,
                padding="max_length",
                max_length=self.models['tokenizer'].model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids
        
        caption_embeds = self.models['text_encoder'](text_input_ids.to(device))
        caption_embeds = caption_embeds[0]

        num_images_per_prompt = 1
        bs_embed, seq_len, _ = caption_embeds.shape
        caption_embeds = caption_embeds.repeat(1, num_images_per_prompt, 1)
        caption_embeds = caption_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return caption_embeds

 
    def get_input(self, batch):
        print(len(batch['caption']))
        z = self.models['vae'].encode(batch["image"]).latent_dist.sample()

        context = self.encode_caption(batch["caption"])

        self.models['scheduler'].set_timesteps(self.config['training_params']['num_timesteps'], device=device)

        timesteps = torch.randint(0, self.models['scheduler'].config.num_train_timesteps, (int(self.config['training_params']['batch_size']),)).to(device)

        grounding_input = prepare_interactdiff_inputs(self.models, batch, int(self.config['training_params']['batch_size']), device)
        
        return z, timesteps, context, grounding_input
    
    def run_one_step(self, batch):
        batch = self.batch_to_device(batch, device)

        x_start, timesteps, context, cross_attention_kwargs = self.get_input(batch)

        self.enable_fuser(True)

        num_channels_latents = self.models['unet'].config.in_channels

        noise = self.prepare_latents(
            self.config['training_params']['batch_size'] * 1,
            num_channels_latents,
            self.height,
            self.width,
            context.dtype,
            device,
            generator = None,
            latents = None,
        )

        x_noisy = self.models['scheduler'].add_noise(original_samples = x_start, noise = noise, timesteps = timesteps)
        x_noisy = self.models['scheduler'].scale_model_input(x_noisy, timesteps)
        
        output = self.models['unet'](x_noisy, timesteps, encoder_hidden_states = context, cross_attention_kwargs = cross_attention_kwargs)
        
        loss = torch.nn.functional.mse_loss(output.sample, noise) * self.l_simple_weight

        return loss
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator=None, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.models['scheduler'].init_noise_sigma
        return latents
    
    def enable_fuser(self, enabled=True):
        for module in self.models['unet'].modules():
            if type(module) is GatedSelfAttentionDense:
                module.enabled = enabled

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
        unet_state_dict = load_file(f"{config}/diffusion_pytorch_model.safetensors")
        unet.load_state_dict(unet_state_dict)
        return unet


    def load_models(self, config):
        unet = self.load_custom_unet(config["unet"])
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=config["vae"]
        ).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path=config["tokenizer"])
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path=config["text_encoder"]).to(device)
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
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "scheduler": scheduler,
        }

        return models


    def train(self):
        for idx in tqdm(range(int(self.config['training_params']['num_iterations']))):
            batch = next(self.train_loader)
            self.batch_to_device(batch, device)

            loss = self.run_one_step(batch)

            #self.scaler.scale(loss).backward()
            #self.scaler.step(self.opt)
            #self.scaler.update()

            loss.backward()
            self.opt.step()
            self.lr_scheduler.step()
            self.opt.zero_grad()

            #print(f"Loss at Iteration {idx} : {loss}")
            print(loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file_path", type=str, required=False, default="./configs/train.yaml"
    )

    args = parser.parse_args()
    trainer = Trainer(args.config_file_path)
    trainer.train()
