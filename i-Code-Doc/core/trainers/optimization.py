from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import SchedulerType
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION


def get_linear_with_fact_schedule_with_warmup(optimizer,
                                              num_warmup_steps,
                                              num_training_steps,
                                              last_epoch=-1,
                                              fact=1.0):
    """Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return fact * float(current_step) / float(max(1, num_warmup_steps))
        return fact * max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(name: Union[str, SchedulerType],
                  optimizer: Optimizer,
                  num_warmup_steps: Optional[int] = None,
                  num_training_steps: Optional[int] = None,
                  fact: Optional[float] = 1.0):
    """Unified API to get any scheduler from its name.

    Args:
        name (:obj:`str` or `:obj:`SchedulerType`):
            The name of the scheduler to use.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (:obj:`int`, `optional`):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (:obj:`int`, `optional`):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        fact (:obj:`float`, `optional`):
            lr fact
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if name == SchedulerType.LINEAR:
        return get_linear_with_fact_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            fact=fact)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f'{name} requires `num_warmup_steps`, please provide that argument.'
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f'{name} requires `num_training_steps`, please provide that argument.'
        )

    return schedule_func(optimizer,
                         num_warmup_steps=num_warmup_steps,
                         num_training_steps=num_training_steps)
