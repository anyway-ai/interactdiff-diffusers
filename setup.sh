#!/bin/bash

# Set Up Conda Environment
conda env create -f environment.yml

# Download Models and Place In Relevant Folders 
wget "https://huggingface.co/interactdiffusion/diffusers-v1-2/resolve/main/text_encoder/model.safetensors" -O "./models/text_encoder/model.safetensors"
wget "https://huggingface.co/interactdiffusion/diffusers-v1-2/resolve/main/unet/diffusion_pytorch_model.safetensors" -O "./models/unet/diffusion_pytorch_model.safetensors"
wget "https://huggingface.co/interactdiffusion/diffusers-v1-2/resolve/main/vae/diffusion_pytorch_model.safetensors" -O "./models/vae/diffusion_pytorch_model.safetensors"

# Download Dataset and Extract into Relevant Folder
mkdir DATA/
wget "https://huggingface.co/datasets/rohanath/hico-clip-det/resolve/main/hico_clip_det_100.tar.gz" -O "./DATA/hico_clip_det_100.tar.gz"
tar -xvzf ./DATA/hico_clip_det_100.tar.gz
mv "./hico_clip_det_100" "./DATA/hico_det_clip"