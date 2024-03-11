"""
https://github.com/lucidrains/make-a-video-pytorch
Duplicate
"""

import math
import functools
from operator import mul

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(val): 
    return val is not None

def default(val, d):
    return val if exists(val) else d

def mul_reduce(tup):
    return functools.reduce(mul, tup)

def divisible_by(numer, denom):
    return (numer % denom) == 0

mlist = nn.ModuleList

# for time conditioning

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert dtype == torch.float, 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)

    
class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        x = (x - mean) * var.clamp(min = eps).rsqrt()
        dtype = self.g.dtype
        return x.to(dtype) * self.g

    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.g


# feedforward
class GEGLU(nn.Module):
    def forward(self, x):
        x = x.float()
        x, gate = x.chunk(2, dim = 1)
        x = x * F.gelu(gate)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.proj_in = nn.Sequential(
            nn.Conv1d(dim, inner_dim * 2, 1, bias = False),
            GEGLU()
        )

        self.proj_out = nn.Sequential(
            ChanLayerNorm(inner_dim),
            nn.Conv1d(inner_dim, dim, 1, bias = False)
        )
        
        nn.init.zeros_(self.proj_out[1].weight)

    def forward(self, x):
        dtype = x.dtype
        x = self.proj_in(x)
        x = self.proj_out(x)
        return x


# feedforwa
# best relative positional encoding

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 1,
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    def forward(self, *dimensions):
        device = self.device

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.to(self.dtype)

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

# helper classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        nn.init.zeros_(self.to_out.weight.data) # identity with skip connection
        
        self.pos_embeds = nn.Parameter(torch.randn([1, 30, dim]))
        self.frame_rate_embeds = nn.Parameter(torch.randn([1, 30, dim]))

    def forward(
        self,
        x,
        context = None,
        rel_pos_bias = None,
        framerate = None,
    ):
        if framerate is not None:
            x = x + self.pos_embeds[:, :x.shape[1]].repeat(x.shape[0], 1, 1)
            
        if context is None:
            context = x
            
        x = self.norm(x)
        context = self.norm(context)

        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return torch.nan_to_num(self.to_out(out))
    
    
class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()   
        self.temporal_attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
        self.temporal_rel_pos_bias = ContinuousPositionBias(dim = dim // 2, heads = heads, num_dims = 1)

        self.ff = FeedForward(dim = dim, mult = 4)
        
    def forward(
        self,
        x,
    ):
        b = x.shape[0]
        time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1])
        x = self.temporal_attn(x, rel_pos_bias = time_rel_pos_bias) + x
        x = self.ff(x.transpose(1, 2)).transpose(1, 2) + x
        return x