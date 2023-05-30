import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvtrans
from core.models import get_model
from core.cfg_helper import model_cfg_bank
from core.common.utils import regularize_image
from einops import rearrange

import pytorch_lightning as pl


class model_module(pl.LightningModule):
    def __init__(self, data_dir='pretrained', pth="model_no_diffusion.pth"):
        super().__init__()
        
        cfgm = model_cfg_bank()('vd_noema')
        cfgm.args.unet_config.args.unet_image_cfg.args.use_video_architecture = True
        cfgm.args.autokl_cfg.map_location = 'cpu'
        cfgm.args.optimus_cfg.map_location = 'cpu'
        cfgm.args.clip_cfg.args.data_dir = data_dir
        
        net = get_model()(cfgm)
        net.load_state_dict(torch.load(os.path.join(data_dir, pth), map_location='cpu'), strict=False)
        print('Load pretrained weight from {}'.format(pth))

        self.net = net
        
        from core.models.ddim_vd import DDIMSampler_VD
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
                    xnew.append(' '.join(xinew))
                x = xnew
            return x
        
        elif xtype == 'audio':
            x = net.audioldm_decode(z)
            x = self.mel_spectrogram_to_waveform(x)
            return x

    def mel_spectrogram_to_waveform(self, mel):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.net.audioldm.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        return waveform
    
    
    def inference(self, xtype, cim=None, ctx=None, cad=None, n_samples=1, mixing=0.3, mixing_c2=0.3, color_adj=None, image_size=256, ddim_steps=50, scale=7.5):
        net = self.net
        sampler = self.sampler
        ddim_eta = 0.0

        first_conditioning = None
        second_conditioning = None
        third_conditioning = None
        first_ctype = None
        second_ctype = None
        third_ctype = None
        if cim is not None:
            ctemp0 = regularize_image(cim).cuda()
            ctemp1 = ctemp0*2 - 1
            ctemp1 = ctemp1[None].repeat(n_samples, 1, 1, 1)
            cim = net.clip_encode_vision(ctemp1).cuda()
            uim = None
            if scale != 1.0:
                dummy = torch.zeros_like(ctemp1).cuda()
                uim = net.clip_encode_vision(dummy).cuda()
            first_conditioning = [uim, cim]
            first_ctype = 'vision'
            
        if cad is not None:
            ctemp = cad[None].repeat(n_samples, 1, 1)
            cad = net.clap_encode_audio(ctemp)
            uad = None
            if scale != 1.0:
                dummy = torch.zeros_like(ctemp)
                uad = net.clap_encode_audio(dummy)  
            if first_conditioning is None:
                first_conditioning = [uad, cad]
                first_ctype = 'audio'
            else:
                second_conditioning = [uad, cad]
                second_ctype = 'audio'
                
        if ctx is not None:        
            ctx = net.clip_encode_text(n_samples * [ctx]).cuda()
            utx = None
            if scale != 1.0:
                utx = net.clip_encode_text(n_samples * [""]).cuda()
            if first_conditioning is None:
                first_conditioning = [utx, ctx]
                first_ctype = 'prompt'    
            elif second_conditioning is None:
                second_conditioning = [utx, ctx]
                second_ctype = 'prompt'
            else:
                third_conditioning = [utx, ctx]
                third_ctype = 'prompt'
        
        
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
            first_conditioning=first_conditioning,
            second_conditioning=second_conditioning,
            third_conditioning=third_conditioning,
            unconditional_guidance_scale=scale,
            xtype=xtype, 
            first_ctype=first_ctype,
            second_ctype=second_ctype,
            third_ctype=third_ctype,
            eta=ddim_eta,
            verbose=False,
            mixed_ratio=mixing, 
            mixed_ratio_c2=mixing_c2)

        out_all = []
        for i, xtype_i in enumerate(xtype):
            z[i] = z[i].cuda()
            x_i = self.decode(z[i], xtype_i)
            out_all.append(x_i)
        return out_all
