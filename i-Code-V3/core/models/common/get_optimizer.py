import torch
import torch.optim as optim
import numpy as np
import itertools

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

class get_optimizer(object):
    def __init__(self):
        self.optimizer = {}
        self.register(optim.SGD, 'sgd')
        self.register(optim.Adam, 'adam')
        self.register(optim.AdamW, 'adamw')

    def register(self, optim, name):
        self.optimizer[name] = optim

    def __call__(self, net, cfg):
        if cfg is None:
            return None
        t = cfg.type
        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
            netm = net.module
        else:
            netm = net
        pg = getattr(netm, 'parameter_group', None)

        if pg is not None:
            params = []
            for group_name, module_or_para in pg.items():
                if not isinstance(module_or_para, list):
                    module_or_para = [module_or_para]

                grouped_params = [mi.parameters() if isinstance(mi, torch.nn.Module) else [mi] for mi in module_or_para]
                grouped_params = itertools.chain(*grouped_params)
                pg_dict = {'params':grouped_params, 'name':group_name}
                params.append(pg_dict)
        else:
            params = net.parameters()
        return self.optimizer[t](params, lr=0, **cfg.args)
