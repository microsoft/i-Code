from typing import List
import os

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from core.models.common.get_model import register
from einops import rearrange

import torchvision.transforms as T

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

from .clip_modules import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig

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
        config = CLIPConfig.from_pretrained(version)
        self.model = CLIPModel(config, add_temporal_attention=False)
        self.max_length = max_length
        self.encode_type = encode_type
        self.fp16 = fp16
        self.freeze()

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
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
        if self.dtype == torch.half:
            tokens = tokens.short()
        outputs = self.model.text_model(input_ids=tokens)
        return outputs.last_hidden_state
        
    def encode_vision_noproj(self, vision_inputs):
        vision_inputs = ((vision_inputs+1)/2)
        # print(vision_inputs.shape)

        transforms = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),]
        )
        if vision_inputs.ndim == 5:
            num_frames = vision_inputs.shape[2]
            vision_inputs = rearrange(vision_inputs, 'b c f h w -> (b f) c h w')
        else:
            num_frames = 1
            vision_inputs = rearrange(vision_inputs, 'b c h w -> b c h w')
        
        pixels = transforms(vision_inputs).to(self.dtype).to(self.device)
        # print(0, vision_inputs.mean(), vision_inputs.var())
        # .to('cpu').numpy()
        
        # if vision_inputs.ndim == 5:
        #     num_frames = vision_inputs.shape[2]
        #     vision_inputs = rearrange(vision_inputs, 'b c f h w -> (b f) h w c')
        # else:
        #     num_frames = 1
        #     vision_inputs = rearrange(vision_inputs, 'b c h w -> b h w c')
            
        # vision_inputs = [vi for vi in vision_inputs]
        # inputs = self.processor(images=vision_inputs, return_tensors="pt")
        # pixels = inputs['pixel_values'].to(self.dtype).to(self.device)
        if num_frames > 1:
            pixels = rearrange(pixels, '(b f) c h w -> b f h w c', f=num_frames)
        # else:
        #     pixels = rearrange(pixels, 'b c h w -> b h w c')    
        outputs = self.model.vision_model(pixel_values=pixels)
        return outputs


    def encode_text(self, text, return_full_seq=False):
        if isinstance(text, List):
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.get_device())
        else:
            tokens = text
        outputs = self.model.text_model(input_ids=tokens)

        if return_full_seq or 1:
            z = self.model.text_projection(outputs.last_hidden_state)
            z_pooled = self.model.text_projection(outputs.pooler_output)
            # print(z.shape)
            z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return z
        else:
            z_pooled = outputs.pooler_output
            z_pooled = self.model.text_projection(z_pooled)
            z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
            return z_pooled.unsqueeze(1)

    def encode_vision(self, images, return_full_seq=False):
        
        outputs = self.encode_vision_noproj(images)

        if return_full_seq:
            z = outputs.last_hidden_state
            z = self.model.vision_model.post_layernorm(z)
            z = self.model.visual_projection(z)
            z_pooled = z[:, 0:1]
            z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
            # print(z.shape)
            return z
        else:
            z_pooled = outputs.pooler_output
            z_pooled = self.model.visual_projection(z_pooled)
            z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
            return z_pooled.unsqueeze(1)

    def encode(self, *args, **kwargs):
        return getattr(self, self.encode_type)(*args, **kwargs)
