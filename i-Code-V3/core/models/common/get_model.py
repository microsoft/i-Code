from email.policy import strict
import torch
import torchvision.models
import os.path as osp
import copy
# from core.common.logger import print_log 
from .utils import \
    get_total_param, get_total_param_sum, \
    get_unit

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def preprocess_model_args(args):
    # If args has layer_units, get the corresponding
    #     units.
    # If args get backbone, get the backbone model.
    args = copy.deepcopy(args)
    if 'layer_units' in args:
        layer_units = [
            get_unit()(i) for i in args.layer_units
        ]
        args.layer_units = layer_units
    if 'backbone' in args:
        args.backbone = get_model()(args.backbone)
    return args

@singleton
class get_model(object):
    def __init__(self):
        self.model = {}
        self.version = {}

    def register(self, model, name, version='x'):
        self.model[name] = model
        self.version[name] = version

    def __call__(self, cfg, verbose=True):
        """
        Construct model based on the config. 
        """
        t = cfg.type

        # the register is in each file
        if t.find('audioldm')==0:
            from ..latent_diffusion.vae import audioldm
        elif t.find('autoencoderkl')==0:
            from ..latent_diffusion.vae import autokl
        elif t.find('optimus')==0:
            from ..latent_diffusion.vae import optimus
            
        elif t.find('clip')==0:
            from ..encoders import clip
        elif t.find('clap')==0:
            from ..encoders import clap   
            
        elif t.find('sd')==0:
            from .. import sd
        elif t.find('codi')==0:
            from .. import codi
        elif t.find('openai_unet')==0:
            from ..latent_diffusion import diffusion_unet
        
        args = preprocess_model_args(cfg.args)
        net = self.model[t](**args)

        return net

    def get_version(self, name):
        return self.version[name]

def register(name, version='x'):
    def wrapper(class_):
        get_model().register(class_, name, version)
        return class_
    return wrapper
