import os
import os.path as osp
import shutil
import copy
import time
import pprint
import numpy as np
import torch
import matplotlib
import argparse
import json
import yaml
from easydict import EasyDict as edict

from core.models import get_model

############
# cfg_bank #
############

def cfg_solvef(cmd, root):
    if not isinstance(cmd, str):
        return cmd
    
    if cmd.find('SAME')==0:
        zoom = root
        p = cmd[len('SAME'):].strip('()').split('.')
        p = [pi.strip() for pi in p]
        for pi in p:
            try:
                pi = int(pi)
            except:
                pass

            try:
                zoom = zoom[pi]
            except:
                return cmd
        return cfg_solvef(zoom, root)

    if cmd.find('SEARCH')==0:
        zoom = root
        p = cmd[len('SEARCH'):].strip('()').split('.')
        p = [pi.strip() for pi in p]
        find = True
        # Depth first search
        for pi in p:
            try:
                pi = int(pi)
            except:
                pass
            
            try:
                zoom = zoom[pi]
            except:
                find = False
                break

        if find:
            return cfg_solvef(zoom, root)
        else:
            if isinstance(root, dict):
                for ri in root:
                    rv = cfg_solvef(cmd, root[ri])
                    if rv != cmd:
                        return rv
            if isinstance(root, list):
                for ri in root:
                    rv = cfg_solvef(cmd, ri)
                    if rv != cmd:
                        return rv
            return cmd

    if cmd.find('MODEL')==0:
        goto = cmd[len('MODEL'):].strip('()')
        return model_cfg_bank()(goto)

    if cmd.find('DATASET')==0:
        goto = cmd[len('DATASET'):].strip('()')
        return dataset_cfg_bank()(goto)

    return cmd

def cfg_solve(cfg, cfg_root):
    # The function solve cfg element such that 
    #   all sorrogate input are settled.
    #   (i.e. SAME(***) ) 
    if isinstance(cfg, list):
        for i in range(len(cfg)):
            if isinstance(cfg[i], (list, dict)):
                cfg[i] = cfg_solve(cfg[i], cfg_root)
            else:
                cfg[i] = cfg_solvef(cfg[i], cfg_root)
    if isinstance(cfg, dict):
        for k in cfg:
            if isinstance(cfg[k], (list, dict)):
                cfg[k] = cfg_solve(cfg[k], cfg_root)
            else:
                cfg[k] = cfg_solvef(cfg[k], cfg_root)        
    return cfg

class model_cfg_bank(object):
    def __init__(self):
        self.cfg_dir = osp.join('configs', 'model')
        self.cfg_bank = edict()
    
    def __call__(self, name):
        if name not in self.cfg_bank:
            cfg_path = self.get_yaml_path(name)
            with open(cfg_path, 'r') as f:
                cfg_new = yaml.load(
                    f, Loader=yaml.FullLoader)
            cfg_new = edict(cfg_new)
            self.cfg_bank.update(cfg_new)

        cfg = self.cfg_bank[name]
        cfg.name = name
        if 'super_cfg' not in cfg:
            cfg = cfg_solve(cfg, cfg)
            self.cfg_bank[name] = cfg
            return copy.deepcopy(cfg)

        super_cfg = self.__call__(cfg.super_cfg)
        # unlike other field,
        # args will not be replaced but update.
        if 'args' in cfg:
            if 'args' in  super_cfg:
                super_cfg.args.update(cfg.args)
            else:
                super_cfg.args = cfg.args
            cfg.pop('args')

        super_cfg.update(cfg)
        super_cfg.pop('super_cfg')
        cfg = super_cfg
        try:
            delete_args = cfg.pop('delete_args')
        except:
            delete_args = []

        for dargs in delete_args:
            cfg.args.pop(dargs)

        cfg = cfg_solve(cfg, cfg)
        self.cfg_bank[name] = cfg
        return copy.deepcopy(cfg)

    def get_yaml_path(self, name):
        if name.find('ldm')==0:
            return osp.join(
                self.cfg_dir, 'ldm.yaml')
        elif name.find('comodgan')==0:
            return osp.join(
                self.cfg_dir, 'comodgan.yaml')
        elif name.find('stylegan')==0:
            return osp.join(
                self.cfg_dir, 'stylegan.yaml')
        elif name.find('absgan')==0:
            return osp.join(
                self.cfg_dir, 'absgan.yaml')
        elif name.find('ashgan')==0:
            return osp.join(
                self.cfg_dir, 'ashgan.yaml')
        elif name.find('sr3')==0:
            return osp.join(
                self.cfg_dir, 'sr3.yaml')
        elif name.find('specdiffsr')==0:
            return osp.join(
                self.cfg_dir, 'specdiffsr.yaml')
        elif name.find('openai_unet')==0:
            return osp.join(
                self.cfg_dir, 'openai_unet.yaml')
        elif name.find('audioldm')==0:
            return osp.join(
                self.cfg_dir, 'audioldm.yaml')
        elif name.find('clip')==0:
            return osp.join(
                self.cfg_dir, 'clip.yaml')
        elif name.find('sd')==0:
            return osp.join(
                self.cfg_dir, 'sd.yaml')
        elif name.find('vd')==0:
            return osp.join(
                self.cfg_dir, 'vd.yaml')
        elif name.find('optimus')==0:
            return osp.join(
                self.cfg_dir, 'optimus.yaml')
        else:
            raise ValueError

class dataset_cfg_bank(object):
    def __init__(self):
        self.cfg_dir = osp.join('configs', 'dataset')
        self.cfg_bank = edict()

    def __call__(self, name):
        if name not in self.cfg_bank:
            cfg_path = self.get_yaml_path(name)
            with open(cfg_path, 'r') as f:
                cfg_new = yaml.load(
                    f, Loader=yaml.FullLoader)
            cfg_new = edict(cfg_new)
            self.cfg_bank.update(cfg_new)

        cfg = self.cfg_bank[name]
        cfg.name = name
        if cfg.get('super_cfg', None) is None:
            cfg = cfg_solve(cfg, cfg)
            self.cfg_bank[name] = cfg
            return copy.deepcopy(cfg)

        super_cfg = self.__call__(cfg.super_cfg)
        super_cfg.update(cfg)
        cfg = super_cfg
        cfg.super_cfg = None
        try:
            delete = cfg.pop('delete')
        except:
            delete = []

        for dargs in delete:
            cfg.pop(dargs)

        cfg = cfg_solve(cfg, cfg)
        self.cfg_bank[name] = cfg
        return copy.deepcopy(cfg)

    def get_yaml_path(self, name):
        if name.find('cityscapes')==0:
            return osp.join(
                self.cfg_dir, 'cityscapes.yaml')
        elif name.find('div2k')==0:
            return osp.join(
                self.cfg_dir, 'div2k.yaml')
        elif name.find('gandiv2k')==0:
            return osp.join(
                self.cfg_dir, 'gandiv2k.yaml')
        elif name.find('srbenchmark')==0:
            return osp.join(
                self.cfg_dir, 'srbenchmark.yaml')
        elif name.find('imagedir')==0:
            return osp.join(
                self.cfg_dir, 'imagedir.yaml')
        elif name.find('places2')==0:
            return osp.join(
                self.cfg_dir, 'places2.yaml')
        elif name.find('ffhq')==0:
            return osp.join(
                self.cfg_dir, 'ffhq.yaml')
        elif name.find('imcpt')==0:
            return osp.join(
                self.cfg_dir, 'imcpt.yaml')
        elif name.find('texture')==0:
            return osp.join(
                self.cfg_dir, 'texture.yaml')
        elif name.find('openimages')==0:
            return osp.join(
                self.cfg_dir, 'openimages.yaml')
        elif name.find('laion2b')==0:
            return osp.join(
                self.cfg_dir, 'laion2b.yaml')
        elif name.find('laionart')==0:
            return osp.join(
                self.cfg_dir, 'laionart.yaml')
        elif name.find('celeba')==0:
            return osp.join(
                self.cfg_dir, 'celeba.yaml')
        elif name.find('coyo')==0:
            return osp.join(
                self.cfg_dir, 'coyo.yaml')
        elif name.find('pafc')==0:
            return osp.join(
                self.cfg_dir, 'pafc.yaml')
        elif name.find('coco')==0:
            return osp.join(
                self.cfg_dir, 'coco.yaml')
        else:
            raise ValueError

class experiment_cfg_bank(object):
    def __init__(self):
        self.cfg_dir = osp.join('configs', 'experiment')
        self.cfg_bank = edict()

    def __call__(self, name):
        if name not in self.cfg_bank:
            cfg_path = self.get_yaml_path(name)
            with open(cfg_path, 'r') as f:
                cfg = yaml.load(
                    f, Loader=yaml.FullLoader)
            cfg = edict(cfg)

        cfg = cfg_solve(cfg, cfg)
        cfg = cfg_solve(cfg, cfg) 
        # twice for SEARCH
        self.cfg_bank[name] = cfg
        return copy.deepcopy(cfg)

    def get_yaml_path(self, name):
        return osp.join(
            self.cfg_dir, name+'.yaml')

def load_cfg_yaml(path):
    if osp.isfile(path):
        cfg_path = path
    elif osp.isfile(osp.join('configs', 'experiment', path)):
        cfg_path = osp.join('configs', 'experiment', path)
    elif osp.isfile(osp.join('configs', 'experiment', path+'.yaml')):
        cfg_path = osp.join('configs', 'experiment', path+'.yaml')
    else:
        assert False, 'No such config!'

    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = edict(cfg)
    cfg = cfg_solve(cfg, cfg)
    cfg = cfg_solve(cfg, cfg)
    return cfg

##############
# cfg_helper #
##############

def get_experiment_id(ref=None):
    if ref is None:
        time.sleep(0.5)
        return int(time.time()*100)
    else:
        try:
            return int(ref)
        except:
            pass
        
        _, ref = osp.split(ref)
        ref = ref.split('_')[0]
        try:
            return int(ref)
        except:
            assert False, 'Invalid experiment ID!'

def record_resume_cfg(path):
    cnt = 0
    while True:
        if osp.exists(path+'.{:04d}'.format(cnt)):
            cnt += 1
            continue
        shutil.copyfile(path, path+'.{:04d}'.format(cnt))
        break

def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--config', type=str)
    parser.add_argument('--gpu', nargs='+', type=int)

    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--addr', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11233)
 
    parser.add_argument('--signature', nargs='+', type=str)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--eval', type=str)
    parser.add_argument('--eval_subdir', type=str)
    parser.add_argument('--pretrained', type=str)

    parser.add_argument('--resume_dir', type=str)
    parser.add_argument('--resume_step', type=int)
    parser.add_argument('--resume_weight', type=str)

    args = parser.parse_args()

    # Special handling the resume
    if args.resume_dir is not None:
        cfg = edict()
        cfg.env = edict()
        cfg.env.debug = args.debug
        cfg.env.resume = edict()
        cfg.env.resume.dir = args.resume_dir
        cfg.env.resume.step = args.resume_step
        cfg.env.resume.weight = args.resume_weight
        return cfg

    cfg = load_cfg_yaml(args.config)
    cfg.env.debug = args.debug
    cfg.env.gpu_device = [0] if args.gpu is None else list(args.gpu)
    cfg.env.master_addr = args.addr
    cfg.env.master_port = args.port
    cfg.env.dist_url = 'tcp://{}:{}'.format(args.addr, args.port)
    cfg.env.node_rank = args.node_rank
    cfg.env.nodes = args.nodes

    istrain = False if args.eval is not None else True
    isdebug = cfg.env.debug

    if istrain:
        if isdebug:
            cfg.env.experiment_id = 999999999999
            cfg.train.signature = ['debug']
        else:
            cfg.env.experiment_id = get_experiment_id()
            if args.signature is not None:
                cfg.train.signature = args.signature
    else:
        if 'train' in cfg:
            cfg.pop('train')
        cfg.env.experiment_id = get_experiment_id(args.eval)
        if args.signature is not None:
            cfg.eval.signature = args.signature

        if isdebug and (args.eval is None):
            cfg.env.experiment_id = 999999999999
            cfg.eval.signature = ['debug']

        if args.eval_subdir is not None:
            if isdebug:
                cfg.eval.eval_subdir = 'debug'
            else:
                cfg.eval.eval_subdir = args.eval_subdir
        if args.pretrained is not None:
            cfg.eval.pretrained = args.pretrained 
          # The override pretrained over the setting in cfg.model

    if args.seed is not None:
        cfg.env.rnd_seed = args.seed

    return cfg

def cfg_initiates(cfg):
    cfge = cfg.env
    isdebug = cfge.debug
    isresume = 'resume' in cfge
    istrain = 'train' in cfg
    haseval = 'eval' in cfg
    cfgt = cfg.train if istrain else None
    cfgv = cfg.eval if haseval else None

    ###############################
    # get some environment params #
    ###############################
    
    cfge.computer = os.uname()
    cfge.torch_version = str(torch.__version__)

    ##########
    # resume #
    ##########

    if isresume:
        resume_cfg_path = osp.join(cfge.resume.dir, 'config.yaml')
        record_resume_cfg(resume_cfg_path)
        with open(resume_cfg_path, 'r') as f:
            cfg_resume = yaml.load(f, Loader=yaml.FullLoader)
        cfg_resume = edict(cfg_resume)
        cfg_resume.env.update(cfge)
        cfg = cfg_resume
        cfge = cfg.env
        log_file = cfg.train.log_file

        print('')
        print('##########')
        print('# resume #')
        print('##########')
        print('')
        with open(log_file, 'a') as f:
            print('', file=f)
            print('##########', file=f)
            print('# resume #', file=f)
            print('##########', file=f)
            print('', file=f)

        pprint.pprint(cfg)
        with open(log_file, 'a') as f:
            pprint.pprint(cfg, f)

    ####################
    # node distributed #
    ####################

    if cfg.env.master_addr!='127.0.0.1':
        os.environ['MASTER_ADDR'] = cfge.master_addr
        os.environ['MASTER_PORT'] = '{}'.format(cfge.master_port)
        if cfg.env.dist_backend=='nccl':
            os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
        if cfg.env.dist_backend=='gloo':
            os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
    
    #######################
    # cuda visible device #
    #######################

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(gid) for gid in cfge.gpu_device]) 

    #####################
    # return resume cfg #
    #####################

    if isresume:
        return cfg

    #############################################
    # some misc setting that not need in resume #
    #############################################

    cfgm = cfg.model
    cfge.gpu_count = len(cfge.gpu_device)

    ##########################################
    # align batch size and num worker config #
    ##########################################
 
    gpu_n = cfge.gpu_count * cfge.nodes
    def align_batch_size(bs, bs_per_gpu): 
        assert (bs is not None) or (bs_per_gpu is not None)
        bs = bs_per_gpu * gpu_n if bs is None else bs
        bs_per_gpu = bs // gpu_n if bs_per_gpu is None else bs_per_gpu
        assert (bs == bs_per_gpu * gpu_n)
        return bs, bs_per_gpu

    if istrain:
        cfgt.batch_size, cfgt.batch_size_per_gpu = \
            align_batch_size(cfgt.batch_size, cfgt.batch_size_per_gpu)
        cfgt.dataset_num_workers, cfgt.dataset_num_workers_per_gpu = \
            align_batch_size(cfgt.dataset_num_workers, cfgt.dataset_num_workers_per_gpu)
    if haseval:
        cfgv.batch_size, cfgv.batch_size_per_gpu = \
            align_batch_size(cfgv.batch_size, cfgv.batch_size_per_gpu)
        cfgv.dataset_num_workers, cfgv.dataset_num_workers_per_gpu = \
            align_batch_size(cfgv.dataset_num_workers, cfgv.dataset_num_workers_per_gpu)

    ##################
    # create log dir #
    ##################

    if istrain:
        if not isdebug:
            sig = cfgt.get('signature', [])
            version = get_model().get_version(cfgm.type)
            sig = sig + ['v{}'.format(version), 's{}'.format(cfge.rnd_seed)]
        else:
            sig = ['debug']

        log_dir = [
            cfge.log_root_dir, 
            '{}_{}'.format(cfgm.symbol, cfgt.dataset.symbol),
            '_'.join([str(cfge.experiment_id)] + sig)
        ]
        log_dir = osp.join(*log_dir)
        log_file = osp.join(log_dir, 'train.log')
        if not osp.exists(log_file):
            os.makedirs(osp.dirname(log_file))
        cfgt.log_dir = log_dir
        cfgt.log_file = log_file

        if haseval:
            cfgv.log_dir = log_dir
            cfgv.log_file = log_file
    else:
        model_symbol = cfgm.symbol
        if cfgv.get('dataset', None) is None:
            dataset_symbol = 'nodataset'
        else:
            dataset_symbol = cfgv.dataset.symbol

        log_dir = osp.join(cfge.log_root_dir, '{}_{}'.format(model_symbol, dataset_symbol))
        exp_dir = search_experiment_folder(log_dir, cfge.experiment_id)
        if exp_dir is None:
            if not isdebug:
                sig = cfgv.get('signature', []) + ['evalonly']
            else:
                sig = ['debug']
            exp_dir = '_'.join([str(cfge.experiment_id)] + sig)

        eval_subdir = cfgv.get('eval_subdir', None)
        # override subdir in debug mode (if eval_subdir is set)
        eval_subdir = 'debug' if (eval_subdir is not None) and isdebug else eval_subdir 

        if eval_subdir is not None:
            log_dir = osp.join(log_dir, exp_dir, eval_subdir)
        else:
            log_dir = osp.join(log_dir, exp_dir)

        disable_log_override = cfgv.get('disable_log_override', False)
        if osp.isdir(log_dir):
            if disable_log_override:
                assert False, 'Override an exsited log_dir is disabled at [{}]'.format(log_dir)
        else:
            os.makedirs(log_dir)

        log_file = osp.join(log_dir, 'eval.log')
        cfgv.log_dir = log_dir
        cfgv.log_file = log_file

    ######################
    # print and save cfg #
    ######################

    pprint.pprint(cfg)
    with open(log_file, 'w') as f:
        pprint.pprint(cfg, f)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    #############
    # save code #
    #############

    save_code = False
    if istrain:
        save_code = cfgt.get('save_code', False)
    elif haseval:
        save_code = cfgv.get('save_code', False)

    if save_code:
        codedir = osp.join(log_dir, 'code')
        if osp.exists(codedir):
            shutil.rmtree(codedir)
        for d in ['configs', 'lib']:
            fromcodedir = d
            tocodedir = osp.join(codedir, d)
            shutil.copytree(
                fromcodedir, tocodedir, 
                ignore=shutil.ignore_patterns(
                    '*__pycache__*', '*build*'))
        for codei in os.listdir('.'):
            if osp.splitext(codei)[1] == 'py':
                shutil.copy(codei, codedir)

    #######################
    # set matplotlib mode #
    #######################

    if 'matplotlib_mode' in cfge:
        try:
            matplotlib.use(cfge.matplotlib_mode)
        except:
            print('Warning: matplotlib mode [{}] failed to be set!'.format(cfge.matplotlib_mode))

    return cfg

def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]) )
        return xnew
    else:
        return x

def search_experiment_folder(root, exid):
    target = None
    for fi in os.listdir(root):
        if not osp.isdir(osp.join(root, fi)):
            continue
        if int(fi.split('_')[0]) == exid:
            if target is not None:
                return None # duplicated
            elif target is None:
                target = fi
    return target
