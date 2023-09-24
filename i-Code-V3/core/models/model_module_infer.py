import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvtrans

from einops import rearrange

import pytorch_lightning as pl

from . import get_model
from ..cfg_helper import model_cfg_bank
from ..common.utils import regularize_image, regularize_video, remove_duplicate_word

import warnings
warnings.filterwarnings("ignore")


class model_module(pl.LightningModule):
    def __init__(self, data_dir='pretrained', pth=["CoDi_encoders.pth"], fp16=False):
        super().__init__()
        
        cfgm = model_cfg_bank()('codi')
        net = get_model()(cfgm)
        if fp16:
            net = net.half()
        for path in pth:
            net.load_state_dict(torch.load(os.path.join(data_dir, path), map_location='cpu'), strict=False)
        print('Load pretrained weight from {}'.format(pth))

        self.net = net
        
        from core.models.ddim.ddim_vd import DDIMSampler_VD
        self.sampler = DDIMSampler_VD(net)

    def decode(self, z, xtype):
        net = self.net
        z = z.cuda()
        if xtype == 'image':
            x = net.autokl_decode(z)
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            x = [tvtrans.ToPILImage()(xi) for xi in x]
            return x
        
        elif xtype == 'video':
            num_frames = z.shape[2]
            z = rearrange(z, 'b c f h w -> (b f) c h w')
            x = net.autokl_decode(z) 
            x = rearrange(x, '(b f) c h w -> b f c h w', f=num_frames)
            
            x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
            video_list = []
            for video in x:
                video_list.append([tvtrans.ToPILImage()(xi) for xi in video])
            return video_list

        elif xtype == 'text':
            prompt_temperature = 1.0
            prompt_merge_same_adj_word = True
            x = net.optimus_decode(z, temperature=prompt_temperature)
            if prompt_merge_same_adj_word:
                xnew = []
                for xi in x:
                    xi_split = xi.split()
                    xinew = []
                    for idxi, wi in enumerate(xi_split):
                        if idxi!=0 and wi==xi_split[idxi-1]:
                            continue
                        xinew.append(wi)
                    xnew.append(remove_duplicate_word(' '.join(xinew)))
                x = xnew
            return x
        
        elif xtype == 'audio':
            x = net.audioldm_decode(z)
            x = net.mel_spectrogram_to_waveform(x)
            return x

    def inference(self, xtype=[], condition=[], condition_types=[], n_samples=1, mix_weight={'video': 1, 'audio': 1, 'text': 1, 'image': 1}, image_size=256, ddim_steps=50, scale=7.5, num_frames=8):
        net = self.net
        sampler = self.sampler
        ddim_eta = 0.0

        conditioning = []
        assert len(set(condition_types)) == len(condition_types), "we don't support condition with same modalities yet."
        assert len(condition) == len(condition_types)
        
        for i, condition_type in enumerate(condition_types):
            if condition_type == 'image':
                ctemp1 = regularize_image(condition[i]).cuda()
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1)
                cim = net.clip_encode_vision(ctemp1).cuda()
                uim = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp1).cuda()
                    uim = net.clip_encode_vision(dummy).cuda()
                conditioning.append(torch.cat([uim, cim]))
                
            elif condition_type == 'video':
                ctemp1 = regularize_video(condition[i]).cuda()
                ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1, 1)
                cim = net.clip_encode_vision(ctemp1).cuda()
                uim = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp1).cuda()
                    uim = net.clip_encode_vision(dummy).cuda()
                conditioning.append(torch.cat([uim, cim]))
                
            elif condition_type == 'audio':
                ctemp = condition[i][None].repeat(n_samples, 1, 1)
                cad = net.clap_encode_audio(ctemp)
                uad = None
                if scale != 1.0:
                    dummy = torch.zeros_like(ctemp)
                    uad = net.clap_encode_audio(dummy)  
                conditioning.append(torch.cat([uad, cad]))
                
            elif condition_type == 'text':
                ctx = net.clip_encode_text(n_samples * [condition[i]]).cuda()
                utx = None
                if scale != 1.0:
                    utx = net.clip_encode_text(n_samples * [""]).cuda()
                conditioning.append(torch.cat([utx, ctx]))
        
        shapes = []
        for xtype_i in xtype:
            if xtype_i == 'image':
                h, w = [image_size, image_size]
                shape = [n_samples, 4, h//8, w//8]
            elif xtype_i == 'video':
                h, w = [image_size, image_size]
                shape = [n_samples, 4, num_frames, h//8, w//8]
            elif xtype_i == 'text':
                n = 768
                shape = [n_samples, n]
            elif xtype_i == 'audio':
                h, w = [256, 16]
                shape = [n_samples, 8, h, w]
            else:
                raise
            shapes.append(shape)
        

        z, _ = sampler.sample(
            steps=ddim_steps,
            shape=shapes,
            condition=conditioning,
            unconditional_guidance_scale=scale,
            xtype=xtype, 
            condition_types=condition_types,
            eta=ddim_eta,
            verbose=False,
            mix_weight=mix_weight)

        out_all = []
        for i, xtype_i in enumerate(xtype):
            z[i] = z[i].cuda()
            x_i = self.decode(z[i], xtype_i)
            out_all.append(x_i)
        return out_all
