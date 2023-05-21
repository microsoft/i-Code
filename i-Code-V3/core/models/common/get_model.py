from email.policy import strict
import torch
import torchvision.models
import os.path as osp
import copy
# from core.common.logger import print_log 
from .utils import \
    get_total_param, get_total_param_sum, \
    get_unit

# def load_state_dict(net, model_path):
#     if isinstance(net, dict):
#         for ni, neti in net.items():
#             paras = torch.load(model_path[ni], map_location=torch.device('cpu'))
#             new_paras = neti.state_dict()
#             new_paras.update(paras)
#             neti.load_state_dict(new_paras)
#     else:
#         paras = torch.load(model_path, map_location=torch.device('cpu'))
#         new_paras = net.state_dict()
#         new_paras.update(paras)
#         net.load_state_dict(new_paras)
#     return

# def save_state_dict(net, path):
#     if isinstance(net, (torch.nn.DataParallel,
#                         torch.nn.parallel.DistributedDataParallel)):
#         torch.save(net.module.state_dict(), path)
#     else:
#         torch.save(net.state_dict(), path)

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
            from .. import audioldm
        elif t=='autoencoderkl':
            from .. import autoencoder
        elif t.find('clip')==0:
            from .. import clip
        elif t.find('sd')==0:
            from .. import sd
        elif t.find('vd')==0:
            from .. import vd
        elif t.find('openai_unet')==0:
            from .. import openaimodel
        elif t.find('optimus')==0:
            from .. import optimus
        args = preprocess_model_args(cfg.args)
        net = self.model[t](**args)

#         map_location = cfg.get('map_location', 'cpu')
#         strict_sd = cfg.get('strict_sd', True)
#         if 'ckpt' in cfg:
#             checkpoint = torch.load(cfg.ckpt, map_location=map_location)
#             net.load_state_dict(checkpoint['state_dict'], strict=strict_sd)
# #             if verbose:
# #                 print_log('Load ckpt from {}'.format(cfg.ckpt))
#         elif 'pth' in cfg:
#             sd = torch.load(cfg.pth, map_location=map_location)
#             net.load_state_dict(sd, strict=strict_sd)
#             if verbose:
#                 print_log('Load pth from {}'.format(cfg.pth))

        # display param_num & param_sum
#         if verbose:
#             print_log(
#                 'Load {} with total {} parameters,' 
#                 '{:.3f} parameter sum.'.format(
#                     t, 
#                     get_total_param(net), 
#                     get_total_param_sum(net) ))

        return net

    def get_version(self, name):
        return self.version[name]

def register(name, version='x'):
    def wrapper(class_):
        get_model().register(class_, name, version)
        return class_
    return wrapper
