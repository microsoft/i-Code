from typing import Dict, List
import os
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import copy
from functools import partial
from contextlib import contextmanager

from .common.get_model import get_model, register
from .sd import DDPM

version = '0'
symbol = 'codi'
    
    
@register('codi', version)
class CoDi(DDPM):
    def __init__(self,
                 audioldm_cfg,
                 autokl_cfg,
                 optimus_cfg,
                 clip_cfg,
                 clap_cfg,
                 vision_scale_factor=0.1812,
                 text_scale_factor=4.3108,
                 audio_scale_factor=0.9228,
                 scale_by_std=False,
                 model_type=[],
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        if 'audioldm' in model_type:
            self.audioldm = get_model()(audioldm_cfg)
        
        if 'autokl' in model_type:
            self.autokl = get_model()(autokl_cfg)
            
        if 'optimus' in model_type:    
            self.optimus = get_model()(optimus_cfg)
        
        if 'clip' in model_type:
            self.clip = get_model()(clip_cfg)
        
        if 'clap' in model_type:
            self.clap = get_model()(clap_cfg)
        
        if not scale_by_std:
            self.vision_scale_factor = vision_scale_factor
            self.text_scale_factor = text_scale_factor
            self.audio_scale_factor = audio_scale_factor
        else:
            self.register_buffer("text_scale_factor", torch.tensor(text_scale_factor))
            self.register_buffer("audio_scale_factor", torch.tensor(audio_scale_factor))
            self.register_buffer('vision_scale_factor', torch.tensor(vision_scale_factor))

        self.freeze()
        
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self.train()
        for param in self.model.parameters():
            param.requires_grad = True
            
    @property
    def device(self):
        return next(self.parameters()).device
            
    @torch.no_grad()
    def autokl_encode(self, image):
        encoder_posterior = self.autokl.encode(image)
        z = encoder_posterior.sample().to(image.dtype)
        return self.vision_scale_factor * z

    @torch.no_grad()
    def autokl_decode(self, z):
        z = 1. / self.vision_scale_factor * z
        return self.autokl.decode(z)

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
        z = 1.0 / self.text_scale_factor * z
        return self.optimus.decode(z, temperature)
    
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
    def mel_spectrogram_to_waveform(self, mel):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.audioldm.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        return waveform
    
    @torch.no_grad()
    def clip_encode_text(self, text, encode_type='encode_text'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip.encode(text)
        self.clip.encode_type = swap_type
        return embedding

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision', return_full_seq=False):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip.encode(vision, return_full_seq)
        self.clip.encode_type = swap_type
        return embedding
    
    @torch.no_grad()
    def clap_encode_audio(self, audio, return_full_seq=False):
        embedding = self.clap(audio, return_full_seq)
        return embedding

    def forward(self, x=None, c=None, noise=None, xtype='image', ctype='prompt', uc=None, x_0=None, u=None, x_0_type='reference', x_0_att_type='concat', return_algined_latents=False):
        if isinstance(x, list):
            t = torch.randint(0, self.num_timesteps, (x[0].shape[0],), device=x[0].device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, noise, xtype, ctype, uc, x_0, u, x_0_type, x_0_att_type, return_algined_latents=return_algined_latents)

    def apply_model(self, x_noisy, t, cond, xtype='image', ctype='prompt', x_0=None, x_0_type='first_frame', return_algined_latents=False):
        return self.model.diffusion_model(x_noisy, t, cond, xtype, ctype, x_0, x_0_type, return_algined_latents=return_algined_latents)

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

    def p_losses(self, x_start, cond, t, noise=None, xtype='image', ctype='prompt', uc=None, x_0=[None], u=None, x_0_type='first_frame', x_0_att_type='concat', return_algined_latents=False):
        noise = [torch.randn_like(x_start_i) for x_start_i in x_start] if noise is None else noise
        x_noisy = [self.q_sample(x_start=x_start_i, t=t, noise=noise_i) for x_start_i, noise_i in zip(x_start, noise)]

        drop_condition = []
        drop_x0 = []
        for j in range(len(noise[0])):
            if random.random() < 0.05:
                drop_condition.append(True)
            else:
                drop_condition.append(False)
            if random.random() < 0.05:
                drop_x0.append(True)
            else:    
                drop_x0.append(False)  
                
        # x_0 is the concatenation condition and does not participate as part of the denoising algorithm
       
        if x_0 is not None:
            if x_0[0] is not None:
                for i in range(len(noise)):
                    if x_0_att_type == 'concat':
                        concat_size = noise[0].shape[2]
                        for j in range(len(u[i])):
                            if drop_x0[j]:
                                x_noisy[i] = torch.cat([u[i], x_noisy[i]], 2)
                            else:
                                noise[i] = torch.cat([x_0[i], noise[i]], 2)
                                x_noisy[i] = torch.cat([x_0[i], x_noisy[i]], 2)
                    elif x_0_att_type == 'cross_att':
                        for j in range(len(u[i])):
                            if drop_x0[j]:
                                x_0[i][j:j+1] = u[i]
                           
            
        if any(drop_condition) and uc is not None:
            for i in range(len(cond)): 
                for j in range(len(uc[i])):
                    if drop_condition[j]:
                        cond[i][j:j+1] = uc[i] 
                    
        model_output = self.apply_model(x_noisy, t, cond, xtype, ctype, x_0, x_0_type, return_algined_latents=return_algined_latents)
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
            if x_0 is not None and x_0_att_type == 'concat':
                if x_0[0] is not None:
                    model_output_i = model_output_i[:, :, -concat_size:]
                    target_i = target_i[:, :, -concat_size:]
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
    