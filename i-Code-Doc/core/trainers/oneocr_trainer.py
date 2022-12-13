import collections
import math
import random
import os
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm import trange
from transformers import (WEIGHTS_NAME, PreTrainedModel, ProgressCallback,
                          TrainerState, is_apex_available, is_comet_available,
                          is_optuna_available, is_ray_available,
                          is_torch_tpu_available, is_wandb_available, logging,
                          set_seed)
from transformers.integrations import AzureMLCallback, hp_params
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, HPSearchBackend,
                                        TrainOutput, speed_metrics, seed_worker)

from core.trainers.data_collator import (DataCollatorForPretraining,
                                  default_data_collator)
from .optimization import get_scheduler
from .trainer_callback import MyAzureMLCallback, MyProgressCallback

from pytorch_lightning.trainer.supporters import CombinedLoader

from PIL import Image


if is_apex_available():
    import apex  # noqa
    from apex import amp  # noqa

_is_native_amp_available = False
if version.parse(torch.__version__) >= version.parse('1.8'):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

if is_wandb_available():
    import wandb  # noqa: F401

if is_comet_available():
    import comet_ml  # noqa: F401

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune  # noqa: F401

logger = logging.get_logger(__name__)

    
class PretrainTrainer(transformers.trainer.Trainer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            samplers = {}
            for key in self.train_dataset:
                samplers[key] = RandomSampler(self.train_dataset[key], generator=generator)                             
            return samplers
        
            return RandomSampler(self.train_dataset, generator=generator)
        else:
            samplers = {}
            for key in self.train_dataset:
                samplers[key] = (RandomSampler(self.train_dataset[key])
                                if self.args.local_rank == -1
                                else DistributedSampler(
                                    self.train_dataset[key],
                                    num_replicas=self.args.world_size,
                                    rank=self.args.process_index,
                                    seed=seed,
                                ))
            return samplers

    def get_train_dataloader(self) -> DataLoader:
        
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            print('is IterableDataset')
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        loaders = {}
        for key in self.train_dataset:
            loaders[key] = DataLoader(
                                self.train_dataset[key],
                                batch_size=self.args.train_batch_size,
                                sampler=train_sampler[key],
                                collate_fn=self.data_collator,
                                drop_last=self.args.dataloader_drop_last,
                                num_workers=self.args.dataloader_num_workers,
                                pin_memory=self.args.dataloader_pin_memory,
                                worker_init_fn=seed_worker,
                            )
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loader

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        
        if self.args.alternate_steps:
            del inputs[random.choice(list(inputs.keys()))]
            
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            output = model(input_dict=inputs)
            
        if self.args.visualize_mae:
            for output_i in output:
                if 'image_output' in output_i.keys():
                    save_visualize_mae(output_i['image_output'], output_i['image_target'], output_i['image_mask_label'], output_dir=self.args.output_dir)
    
        loss = sum([output_i['loss'] for output_i in output])
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()

    
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        '''
        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            if smp.dp_rank() == 0:
                # Consolidate the state dict on all processed of dp_rank 0
                opt_state_dict = self.optimizer.state_dict()
                # Save it and the scheduler on the main process
                if self.args.should_save:
                    torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
                    if self.use_amp:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.use_amp:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        '''

        # Determine the new best metric / best model checkpoint
        '''
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
        '''
        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)



def _model_unwrap(model: nn.Module) -> nn.Module:
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, 'module'):
        return _model_unwrap(model.module)
    else:
        return model
