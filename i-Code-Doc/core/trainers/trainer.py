import random
import os
import numpy as np
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import transformers
from transformers import logging
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker

from pytorch_lightning.trainer.supporters import CombinedLoader

from PIL import Image

logger = logging.get_logger(__name__)

    
def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 16
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def save_visualize_mae(image_output_, image_target, image_mask_label=None, output_dir=None):
    save_dir = os.path.join(output_dir, 'mae_results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    image_output = unpatchify(image_output_)
    image_output_[~image_mask_label.bool()] = 0.8
    image_output_masked = unpatchify(image_output_)
    if len(image_target.shape) == 3:
        image_target = unpatchify(image_target)

    for k in range(len(image_output)):
        im_o = np.transpose(np.clip(image_output[k].float().detach().cpu().numpy()*255.0, 0.0, 255.0).astype(np.uint8), (1,2,0))
        im_t = np.transpose(np.clip(image_target[k].float().detach().cpu().numpy()*255.0, 0.0, 255.0).astype(np.uint8), (1,2,0))
        im_o_masked = np.transpose(np.clip(image_output_masked[k].float().detach().cpu().numpy()*255.0, 0.0, 255.0).astype(np.uint8), (1,2,0))
        im_concat = np.concatenate([im_o, im_o_masked, im_t], 1)
        Image.fromarray(im_concat).save(os.path.join(save_dir, f'concat_{str(k)}.jpg'))

        Image.fromarray(im_o).save(os.path.join(save_dir, f'output_{str(k)}.jpg'))
        Image.fromarray(im_t).save(os.path.join(save_dir, f'target_{str(k)}.jpg'))
        Image.fromarray(im_o_masked).save(os.path.join(save_dir, f'output_masked_{str(k)}.jpg'))
    
    
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
