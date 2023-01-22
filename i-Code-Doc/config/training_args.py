import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.file_utils import (cached_property, is_torch_tpu_available,
                                     torch_required)
from transformers.training_args import TrainingArguments
from transformers.utils import logging

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
logger = logging.get_logger(__name__)


@dataclass
class UdopTrainingArguments(TrainingArguments):
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'When using distributed training, the value of the flag `find_unused_parameters` passed to '
            '`DistributedDataParallel`.'
        },
    )
    profile: bool = field(default=False,
                          metadata={'help': 'whether to enable profiling'})
    optimizer: str = field(
        default='transformers_AdamW',
        metadata={
            'help':
            "should be chosen in [\"transformers_AdamW\", \"torch_AdamW\", \"apex_FusedAdam\", \"apex_FusedLAMB\"]"
        })
    continue_training: bool = field(
        default=False,
        metadata={
            'help': 'Set `True` to load training state (default: `False`)'
        })

    prefetch_factor: int = field(default=2)

    dataloader_timeout: int = field(default=0)

    lr_fact: float = field(default=1.0)

    @cached_property
    @torch_required
    def _setup_devices(self) -> 'torch.device':
        logger.info('PyTorch: setting up devices')
        if self.no_cuda:
            logger.info('runnning on cpu')
            device = torch.device('cpu')
            self._n_gpu = 0
        elif is_torch_tpu_available():
            logger.info('runnning on tpu')
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            logger.info('runnning with dp')
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            logger.info('runnning with ddp')
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                logger.info('runnning with deepspeed')
                from transformers.integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError(
                        '--deepspeed requires deepspeed: `pip install deepspeed`.'
                    )
                import deepspeed

                deepspeed.init_distributed()
            else:
                backend = None
                if backend is None:
                    if torch.distributed.is_nccl_available():
                        backend = 'nccl'
                    elif torch.distributed.is_mpi_available():
                        backend = 'mpi'
                    else:
                        backend = 'gloo'
                logger.info('init process group using backend ' + backend)
                torch.distributed.init_process_group(
                    backend=backend, timeout=datetime.timedelta(days=1))
            device = torch.device('cuda', self.local_rank)
            self._n_gpu = 1

        if device.type == 'cuda':
            torch.cuda.set_device(device)

        return device

