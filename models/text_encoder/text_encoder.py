import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        from transformers import logging as tl
        tl.set_verbosity_error()  # ignore the annoying warning when using CLIPTextModel
        self.transformer = CLIPTextModel.from_pretrained(version)
        tl.set_verbosity_warning()
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_pooler_output=False):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        
        if not return_pooler_output:
            return z
        else:
            return z, outputs.pooler_output

    def encode(self, text, return_pooler_output=False):
        return self(text, return_pooler_output)