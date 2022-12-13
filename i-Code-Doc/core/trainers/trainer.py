import random
import os
from typing import Any, Dict, Optional, Union

from packaging import version

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler
import transformers
from transformers import is_apex_available, logging, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker

from pytorch_lightning.trainer.supporters import CombinedLoader


logger = logging.get_logger(__name__)

    
if is_apex_available():
    import apex
    from apex import amp

    
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

        run_dir = self.args.output_dir
        self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)



def _model_unwrap(model: nn.Module) -> nn.Module:
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, 'module'):
        return _model_unwrap(model.module)
    else:
        return model
