---
license: bsd
---

# InteractDiffusion Diffuser Implementation

[Project Page](https://jiuntian.github.io/interactdiffusion) |
 [Paper](https://arxiv.org/abs/2312.05849) |
 [WebUI](https://github.com/jiuntian/sd-webui-interactdiffusion) |
 [Demo](https://huggingface.co/spaces/interactdiffusion/interactdiffusion) |
 [Video](https://www.youtube.com/watch?v=Uunzufq8m6Y) |
 [Diffuser](https://huggingface.co/interactdiffusion/diffusers-v1-2) |
 [Colab](https://colab.research.google.com/drive/1Bh9PjfTylxI2rbME5mQJtFqNTGvaghJq?usp=sharing)

## How to Use

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "interactdiffusion/diffusers-v1-2",
    trust_remote_code=True,
    variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

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

images[0].save('out.jpg')
```

For more information, please check the [project homepage](https://jiuntian.github.io/interactdiffusion/).

## Citation

```bibtex
@inproceedings{hoe2023interactdiffusion,
      title={InteractDiffusion: Interaction Control in Text-to-Image Diffusion Models}, 
      author={Jiun Tian Hoe and Xudong Jiang and Chee Seng Chan and Yap-Peng Tan and Weipeng Hu},
      year={2024},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## Acknowledgement

This work is developed based on the codebase of [GLIGEN](https://github.com/gligen/GLIGEN) and [LDM](https://github.com/CompVis/latent-diffusion).
