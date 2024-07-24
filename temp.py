import torch
from diffusers import DDIMScheduler, AutoencoderKL, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from unet.interactdiffusion_unet_2d_condition import InteractDiffusionUNet2DConditionModel

from pipeline_stable_diffusion_interactdiffusion import StableDiffusionInteractDiffusionPipeline


scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path = "./scheduler")
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = "./vae", variant = "fp16")
text_encoder = CLIPTextModel.from_pretrained("./text_encoder", variant = "fp16")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = "./tokenizer")
unet = InteractDiffusionUNet2DConditionModel.from_pretrained("./unet", variant = "fp16")


model0 = StableDiffusionInteractDiffusionPipeline(vae = vae, 
                                                 text_encoder=text_encoder, 
                                                 tokenizer=tokenizer, 
                                                 unet=unet, 
                                                 scheduler = scheduler, 
                                                 safety_checker=None, 
                                                 feature_extractor=None)

model0 = model0.to('cuda')
images = model0(
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

images[0].save('out0.jpg')




model1 = DiffusionPipeline.from_pretrained(
    "interactdiffusion/diffusers-v1-2",
    trust_remote_code=True,
    variant="fp16", torch_dtype=torch.float16
)

model1 = model1.to('cuda')
images = model1(
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

images[0].save('out1.jpg')
