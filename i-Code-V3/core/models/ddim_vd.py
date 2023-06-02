"""
https://github.com/SHI-Labs/Versatile-Diffusion
"""


import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from .ddim import DDIMSampler

class DDIMSampler_VD(DDIMSampler):
    @torch.no_grad()
    def sample(self,
               steps,
               shape,
               xt=None,
               condition=None,
               unconditional_guidance_scale=1.,
               xtype='image',
               condition_types=['text'],
               eta=0.,
               temperature=1.,
               mix_weight=None,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100,):

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
        print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self.ddim_sampling(
            shape,
            xt=xt,
            condition=condition,
            unconditional_guidance_scale=unconditional_guidance_scale,
            xtype=xtype,
            condition_types=condition_types,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            log_every_t=log_every_t,
            mix_weight=mix_weight,)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, 
                      shape,
                      xt=None,
                      condition=None,
                      unconditional_guidance_scale=1., 
                      xtype=['image'],
                      condition_types=['text'],
                      ddim_use_original_steps=False,
                      timesteps=None, 
                      noise_dropout=0., 
                      temperature=1.,
                      mix_weight=None,
                      log_every_t=100,):

        device = self.model.device
        dtype = condition[0][0].dtype
        
        if isinstance(shape[0], list):
            bs = shape[0][0]
        else:    
            bs = shape[0]
        if xt is None:
            if isinstance(shape[0], list):
                xt = [torch.randn(shape_i, device=device, dtype=dtype) for shape_i in shape]
            else:    
                xt = torch.randn(shape, device=device, dtype=dtype)
                
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'pred_xt': [], 'pred_x0': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        pred_xt = xt
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((bs,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(
                pred_xt, 
                condition, 
                ts, index, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                xtype=xtype,
                condition_types=condition_types,
                use_original_steps=ddim_use_original_steps,
                noise_dropout=noise_dropout,
                temperature=temperature,
                mix_weight=mix_weight,)
            pred_xt, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['pred_xt'].append(pred_xt)
                intermediates['pred_x0'].append(pred_x0)

        return pred_xt, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, 
                      condition,
                      t, index, 
                      unconditional_guidance_scale=1., 
                      xtype=['image'],
                      condition_types=['text'],
                      repeat_noise=False, 
                      use_original_steps=False, 
                      noise_dropout=0.,
                      temperature=1.,
                      mix_weight=None,):

        b, *_, device = *x[0].shape, x[0].device

        x_in = []
        for x_i in x:
            x_in.append(torch.cat([x_i] * 2))
        t_in = torch.cat([t] * 2)
        
        out = self.model.model.diffusion_model(
            x_in, t_in, condition, xtype=xtype, condition_types=condition_types, mix_weight=mix_weight)
        e_t = []
        for out_i in out:
            e_t_uncond_i, e_t_i = out_i.chunk(2)
            e_t_i = e_t_uncond_i + unconditional_guidance_scale * (e_t_i - e_t_uncond_i)
            e_t_i = e_t_i.to(device)
            e_t.append(e_t_i)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        x_prev = []
        pred_x0 = []
        device = x[0].device
        dtype = x[0].dtype
        for i, xtype_i in enumerate(xtype):
            if xtype_i == 'image':
                extended_shape = (b, 1, 1, 1)
            elif xtype_i == 'video':
                extended_shape = (b, 1, 1, 1, 1)        
            elif xtype_i == 'text':
                extended_shape = (b, 1)
            elif xtype_i == 'audio':
                extended_shape = (b, 1, 1, 1)    

            a_t = torch.full(extended_shape, alphas[index], device=device, dtype=dtype)
            a_prev = torch.full(extended_shape, alphas_prev[index], device=device, dtype=dtype)
            sigma_t = torch.full(extended_shape, sigmas[index], device=device, dtype=dtype)
            sqrt_one_minus_at = torch.full(extended_shape, sqrt_one_minus_alphas[index], device=device, dtype=dtype)

            # current prediction for x_0
            pred_x0_i = (x[i] - sqrt_one_minus_at * e_t[i]) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t[i]
            noise = sigma_t * noise_like(x[i], repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev_i = a_prev.sqrt() * pred_x0_i + dir_xt + noise
            x_prev.append(x_prev_i)
            pred_x0.append(pred_x0_i)
        return x_prev, pred_x0