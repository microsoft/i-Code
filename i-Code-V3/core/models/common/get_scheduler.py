import torch
import torch.optim as optim
import numpy as np
import copy
from ... import sync
from ...cfg_holder import cfg_unique_holder as cfguh

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class get_scheduler(object):
    def __init__(self):
        self.lr_scheduler = {}

    def register(self, lrsf, name):
        self.lr_scheduler[name] = lrsf

    def __call__(self, cfg):
        if cfg is None:
            return None
        if isinstance(cfg, list):
            schedulers = []
            for ci in cfg:
                t = ci.type
                schedulers.append(
                    self.lr_scheduler[t](**ci.args))
            if len(schedulers) == 0:
                raise ValueError
            else:
                return compose_scheduler(schedulers)
        t = cfg.type
        return self.lr_scheduler[t](**cfg.args)
        

def register(name):
    def wrapper(class_):
        get_scheduler().register(class_, name)
        return class_
    return wrapper

class template_scheduler(object):
    def __init__(self, step):
        self.step = step

    def __getitem__(self, idx):
        raise ValueError

    def set_lr(self, optim, new_lr, pg_lrscale=None):
        """
        Set Each parameter_groups in optim with new_lr
        New_lr can be find according to the idx.
        pg_lrscale tells how to scale each pg.
        """
        # new_lr = self.__getitem__(idx)
        pg_lrscale = copy.deepcopy(pg_lrscale)
        for pg in optim.param_groups:
            if pg_lrscale is None:
                pg['lr'] = new_lr
            else:
                pg['lr'] = new_lr * pg_lrscale.pop(pg['name'])
        assert (pg_lrscale is None) or (len(pg_lrscale)==0), \
            "pg_lrscale doesn't match pg"

@register('constant')
class constant_scheduler(template_scheduler):
    def __init__(self, lr, step):
        super().__init__(step)
        self.lr = lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr

@register('poly')
class poly_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, power, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b = self.start_lr, self.end_lr
        p, n = self.power, self.step
        return b + (a-b)*((1-idx/n)**p)

@register('linear')
class linear_scheduler(template_scheduler):
    def __init__(self, start_lr, end_lr, step):
        super().__init__(step)
        self.start_lr = start_lr
        self.end_lr = end_lr

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        a, b, n = self.start_lr, self.end_lr, self.step
        return b + (a-b)*(1-idx/n)

@register('multistage')
class constant_scheduler(template_scheduler):
    def __init__(self, start_lr, milestones, gamma, step):
        super().__init__(step)
        self.start_lr = start_lr
        m = [0] + milestones + [step]
        lr_iter = start_lr
        self.lr = []
        for ms, me in zip(m[0:-1], m[1:]):
            for _ in range(ms, me):
                self.lr.append(lr_iter)
            lr_iter *= gamma

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        return self.lr[idx]

class compose_scheduler(template_scheduler):
    def __init__(self, schedulers):
        self.schedulers = schedulers
        self.step = [si.step for si in schedulers]
        self.step_milestone = []
        acc = 0
        for i in self.step:
            acc += i
            self.step_milestone.append(acc)
        self.step = sum(self.step)

    def __getitem__(self, idx):
        if idx >= self.step:
            raise ValueError
        ms = self.step_milestone
        for idx, (mi, mj) in enumerate(zip(ms[:-1], ms[1:])):
            if mi <= idx < mj:
                return self.schedulers[idx-mi]
        raise ValueError

####################
# lambda schedular #
####################

class LambdaWarmUpCosineScheduler(template_scheduler):
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, 
                 base_lr,
                 warm_up_steps, 
                 lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        cfgt = cfguh().cfg.train
        bs = cfgt.batch_size
        if 'gradacc_every' not in cfgt:
            print('Warning, gradacc_every is not found in xml, use 1 as default.')
        acc = cfgt.get('gradacc_every', 1)
        self.lr_multi = base_lr * bs * acc
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __getitem__(self, idx):
        return self.schedule(idx) * self.lr_multi

class LambdaWarmUpCosineScheduler2(template_scheduler):
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """
    def __init__(self, 
                 base_lr,
                 warm_up_steps, 
                 f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        cfgt = cfguh().cfg.train
        # bs = cfgt.batch_size
        # if 'gradacc_every' not in cfgt:
        #     print('Warning, gradacc_every is not found in xml, use 1 as default.')
        # acc = cfgt.get('gradacc_every', 1)
        # self.lr_multi = base_lr * bs * acc
        self.lr_multi = base_lr
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __getitem__(self, idx):
        return self.schedule(idx) * self.lr_multi

@register('stable_diffusion_linear')
class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                      f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f