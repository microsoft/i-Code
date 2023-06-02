from typing import List
import os

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from core.models.common.get_model import register
from einops import rearrange

version = '0'
symbol = 'clip'

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

from transformers import CLIPTokenizer, CLIPTextModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

@register('clip_text_frozen', version)
class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

from core.models.transformers_clip import CLIPProcessor, CLIPModel, CLIPTokenizer

@register('clip_frozen', version)
class FrozenCLIP(AbstractEncoder):
    def __init__(self, 
                 version="openai/clip-vit-large-patch14", 
                 max_length=77, 
                 encode_type='encode_text',
                 fp16=False, 
                 data_dir='.'):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.model = CLIPModel.from_pretrained(version, add_temporal_attention=False)
        self.max_length = max_length
        self.encode_type = encode_type
        self.fp16 = fp16
#         self.freeze()

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self):
        self.model = self.model.eval()
        self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def encode_text_pooled(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.get_text_features(input_ids=tokens)
        return outputs

    def encode_vision_pooled(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].half() if self.fp16 else inputs['pixel_values']
        pixels = pixels.to(self.get_device())
        return self.model.get_image_features(pixel_values=pixels)

    def encode_text_noproj(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.text_model(input_ids=tokens)
        return outputs.last_hidden_state
        
    def encode_vision_noproj(self, images, num_frames=1):
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].to(self.dtype).to(self.get_device())
        if num_frames > 1:
            pixels = rearrange(pixels, '(b f) h w c -> b f h w c', f=num_frames)
        outputs = self.model.vision_model(pixel_values=pixels)
        return outputs


    def encode_text(self, text):
        if isinstance(text, List):
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.get_device())
        else:
            tokens = text
        outputs = self.model.text_model(input_ids=tokens)
        z_pooled = outputs.pooler_output
        z_pooled = self.model.text_projection(z_pooled)
        z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z_pooled.unsqueeze(1)

    def encode_vision(self, images, num_frames=1):
        z = self.encode_vision_noproj(images, num_frames)
        z_pooled = z.pooler_output
        z_pooled = self.model.visual_projection(z_pooled)
        z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z_pooled.unsqueeze(1)

    def encode(self, *args, **kwargs):
        return getattr(self, self.encode_type)(*args, **kwargs)
