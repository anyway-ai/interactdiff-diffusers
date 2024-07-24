from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from unet.interactdiffusion_unet_2d_condition import InteractDiffusionUNet2DConditionModel

from pipeline_stable_diffusion_interactdiffusion import StableDiffusionInteractDiffusionPipeline


scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path = "./scheduler")
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = "./vae", variant = "fp16")
text_encoder = CLIPTextModel.from_pretrained("./text_encoder", variant = "fp16")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path = "./tokenizer")
unet = InteractDiffusionUNet2DConditionModel.from_pretrained("./unet", variant = "fp16")


model = StableDiffusionInteractDiffusionPipeline(vae = vae, 
                                                 text_encoder=text_encoder, 
                                                 tokenizer=tokenizer, 
                                                 unet=unet, 
                                                 scheduler = scheduler, 
                                                 safety_checker=False, 
                                                 feature_extractor=None)

