"""
https://github.com/SHI-Labs/Versatile-Diffusion
"""

from typing import Dict, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import copy
from functools import partial
from contextlib import contextmanager
from core.models.common.get_model import get_model, register

version = '0'
symbol = 'vd'

from .diffusion_utils import \
    count_params, extract_into_tensor, make_beta_schedule
from .distributions import normal_kl, DiagonalGaussianDistribution

from .audio_autoencoder import AudioAutoencoderKL
from .clap import CLAPAudioEmbeddingClassifierFreev2
from .autoencoder import AutoencoderKL

from .sd import DDPM
    
    
@register('vd', version)
class VD(DDPM):
    def __init__(self,
                 autokl_cfg,
                 optimus_cfg,
                 clip_cfg,
                 scale_factor=1.0,
                 text_scale_factor=4.3108,
                 audio_scale_factor=0.9228,
                 scale_by_std=False,
                 *args, 
                 **kwargs):
        self.scale_by_std = scale_by_std
        super().__init__(*args, **kwargs)
        
        self.max_text_len = autokl_cfg
        self.audioldm = AudioAutoencoderKL()
        self.audioldm.eval()
        for param in self.audioldm.parameters():
            param.requires_grad = False
            
        self.clap = CLAPAudioEmbeddingClassifierFreev2(joint_embed_shape=768, amodel="HTSAT-large")
        self.clap.eval()
        for param in self.clap.parameters():
            param.requires_grad = False
        
        self.autokl = get_model()(autokl_cfg)
        self.autokl.eval()
        for param in self.autokl.parameters():
            param.requires_grad = False
            
        self.optimus = get_model()(optimus_cfg)
        self.optimus.eval()
        for param in self.optimus.parameters():
            param.requires_grad = False
            
        self.clip = get_model()(clip_cfg)
        self.clip.eval()
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.device = 'cpu'
        
        self.concat_mode = 'crossattn'
        if not scale_by_std:
            self.image_scale_factor = scale_factor
            self.text_scale_factor = text_scale_factor
            self.audio_scale_factor = audio_scale_factor
        else:
            self.register_buffer("text_scale_factor", torch.tensor(scale_factor))
            self.register_buffer("audio_scale_factor", torch.tensor(audio_scale_factor))
            self.register_buffer('image_scale_factor', torch.tensor(scale_factor))

    @torch.no_grad()
    def autokl_encode(self, image):
        encoder_posterior = self.autokl.encode(image)
        z = encoder_posterior.sample().to(image.dtype)
        return self.image_scale_factor * z

    @torch.no_grad()
    def autokl_decode(self, z):
        z = 1. / self.image_scale_factor * z
        return self.autokl.decode(z)

    def mask_tokens(inputs, tokenizer, args):
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        
        masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).to(torch.uint8)
        labels[masked_indices==1] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.uint8) & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
        indices_random = indices_random
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    @torch.no_grad()
    def optimus_encode(self, text):
        if isinstance(text, List):
            tokenizer = self.optimus.tokenizer_encoder
            token = [tokenizer.tokenize(sentence.lower()) for sentence in text]
            token_id = []
            for tokeni in token:
                token_sentence = [tokenizer._convert_token_to_id(i) for i in tokeni]
                token_sentence = tokenizer.add_special_tokens_single_sentence(token_sentence)
                token_id.append(torch.LongTensor(token_sentence))
            token_id = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=0.0)[:, :512]
        else:
            token_id = text
        z = self.optimus.encoder(token_id, attention_mask=(token_id > 0))[1]
        z_mu, z_logvar = self.optimus.encoder.linear(z).chunk(2, -1)
        return z_mu.squeeze(1) * self.text_scale_factor

    @torch.no_grad()
    def optimus_decode(self, z, temperature=1.0):
        bos_token = self.optimus.tokenizer_decoder.encode('<BOS>')
        eos_token = self.optimus.tokenizer_decoder.encode('<EOS>')
        context_tokens = torch.LongTensor(bos_token).to(z.device)

        from .optimus import sample_single_sequence_conditional
        sentenses = []
        for zi in z:
            scaled_zi = 1.0 / self.text_scale_factor * zi
            out = sample_single_sequence_conditional(
                model=self.optimus.decoder,
                context=context_tokens,
                past=scaled_zi, temperature=temperature, 
                top_k=0, top_p=1.0,
                max_length=30,
                eos_token = eos_token[0],)
            text = self.optimus.tokenizer_decoder.decode(out.tolist(), clean_up_tokenization_spaces=True)
            text = text.split()[1:-1]
            text = ' '.join(text)
            sentenses.append(text)
        return sentenses
    
    @torch.no_grad()
    def audioldm_encode(self, audio, time=2.0):
        encoder_posterior = self.audioldm.encode(audio, time=time)
        z = encoder_posterior.sample().to(audio.dtype)
        return z * self.audio_scale_factor

    @torch.no_grad()
    def audioldm_decode(self, z):
        if (torch.max(torch.abs(z)) > 1e2):
            z = torch.clip(z, min=-10, max=10)
        z = 1.0 / self.audio_scale_factor * z
        return self.audioldm.decode(z)
    
    @torch.no_grad()
    def clip_encode_text(self, text, encode_type='encode_text'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip.encode(text)
        self.clip.encode_type = swap_type
        return embedding

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        if isinstance(vision, torch.Tensor):
            if vision.ndim == 5:
                num_frames = vision.shape[2]
                vision = rearrange(vision, 'b c f h w -> (b f) c h w')
            else:
                num_frames = 1
            vision = ((vision+1)/2).to('cpu').numpy()
            vision = np.transpose(vision, (0, 2, 3, 1))
            vision = [vi for vi in vision]
        embedding = self.clip.encode(vision, num_frames=num_frames)
        self.clip.encode_type = swap_type
        return embedding
    
    @torch.no_grad()
    def clap_encode_audio(self, audio):
        embedding = self.clap(audio)
        return embedding

    def forward(self, x=None, c=None, noise=None, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        if isinstance(x, list):
            t = torch.randint(0, self.num_timesteps, (x[0].shape[0],), device=x[0].device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, noise, xtype, ctype, u, return_algined_latents)

    def apply_model(self, x_noisy, t, cond, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        return self.model.diffusion_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents)

    def get_pixel_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=-0.0)
        return loss

    def get_text_loss(self, pred, target):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)    
        return loss

    def p_losses(self, x_start, cond, t, noise=None, xtype='image', ctype='prompt', u=None, return_algined_latents=False):
        if isinstance(x_start, list):
            noise = [torch.randn_like(x_start_i) for x_start_i in x_start] if noise is None else noise
            x_noisy = [self.q_sample(x_start=x_start_i, t=t, noise=noise_i) for x_start_i, noise_i in zip(x_start, noise)]
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents)
            if return_algined_latents:
                return model_output
            
            loss_dict = {}

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()
            
            loss = 0.0
            for model_output_i, target_i, xtype_i in zip(model_output, target, xtype):
                if xtype_i == 'image':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                elif xtype_i == 'video':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3, 4])
                elif xtype_i == 'text':
                    loss_simple = self.get_text_loss(model_output_i, target_i).mean([1])
                elif xtype_i == 'audio':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                loss += loss_simple.mean()
            return loss / len(xtype)
        
        else:    
            noise = torch.randn_like(x_start) if noise is None else noise
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype)

            loss_dict = {}

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()

            if xtype == 'image':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            elif xtype == 'video':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
            elif xtype == 'text':
                loss_simple = self.get_text_loss(model_output, target).mean([1])
            elif xtype == 'audio':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss = loss_simple.mean()
            return loss