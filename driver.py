import json

import torch
from diffusers import DDIMScheduler, AutoencoderKL, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from unet.unet import InteractDiffusionUNet2DConditionModel

from pipeline import StableDiffusionInteractDiffusionPipeline

def test_pipeline(pipeline, out_name):
    images = pipeline(
    prompt="a person is feeding a cat",
    interactdiffusion_subject_phrases=["person"],
    interactdiffusion_object_phrases=["cat"],
    interactdiffusion_action_phrases=["feeding"],
    interactdiffusion_subject_boxes=[[0.0332, 0.1660, 0.3359, 0.7305]],
    interactdiffusion_object_boxes=[[0.2891, 0.4766, 0.6680, 0.7930]],
    interactdiffusion_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
    ).images

    images[0].save(f'./outputs/{out_name}.jpg')

#scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path = "./scheduler")
#vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = "./vae", variant = "fp16")
#text_encoder = CLIPTextModel.from_pretrained("./text_encoder", variant = "fp16")
#tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = "./tokenizer")

with open("./unet/config.json", 'r') as f:
    unet_configs = json.load(f)

del unet_configs["_class_name"]
del unet_configs["_diffusers_version"]

unet = InteractDiffusionUNet2DConditionModel(**unet_configs)
print(unet.keys())

"""pipeline = StableDiffusionInteractDiffusionPipeline(vae = vae, 
                                                 text_encoder=text_encoder, 
                                                 tokenizer=tokenizer, 
                                                 unet=unet, 
                                                 scheduler = scheduler, 
                                                 safety_checker=None, 
                                                 feature_extractor=None)

pipeline = pipeline.to('cuda')

test_pipeline(pipeline, "out3")"""

#unet = InteractDiffusionUNet2DConditionModel.from_pretrained("./unet", variant = "fp16")