import argparse
import logging

import pytorch_lightning as pl

from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from product_key_memory import PKM
from transformers.optimization import AdamW


logger = logging.getLogger(__name__)

def get_linear_schedule_with_warmup_until_half(*args, **kwargs):
    """
    Minor modification of linear scheduler. LR doesn't drop to the 0 at the end of the training
    instead, it drops until half of the value of peak learning rate.
    """
    num_training_steps = kwargs.pop("num_training_steps") * 2
    return get_linear_schedule_with_warmup(*args, num_training_steps=num_training_steps, **kwargs)


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "linear_until_half": get_linear_schedule_with_warmup_until_half,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}

arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading

        self.save_hyperparameters(hparams)
        self.step_count = 0

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = get_schedule_func(self.opt, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "cosine_w_restarts":
            num_cycles = self.hparams.max_epochs
            scheduler = get_schedule_func(
                self.opt,
                num_warmup_steps=self.hparams.warmup_steps,
                num_cycles=num_cycles,
                num_training_steps=self.total_steps(),
            )
        else:
            scheduler = get_schedule_func(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
            )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def fetch_pkm_value_parameters(self, module):
        params = []
        for m in module.modules():
            if isinstance(m, PKM):
                params.extend([p for p, _ in m.named_parameters()])
        return params

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model

        no_decay = ["bias", "LayerNorm.weight"]
        pkm_parameters = self.fetch_pkm_value_parameters(model)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and n not in pkm_parameters
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and not n in pkm_parameters
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if n in pkm_parameters],
                "lr": self.hparams.pkm_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        if self.hparams.max_steps:
            return self.hparams.max_steps
        ngpus = self.hparams.gpus
        if isinstance(ngpus, list):
            ngpus = len(ngpus)
        elif isinstance(ngpus, str):
            ngpus = len([a for a in ngpus.split(",") if a.strip()])
        num_devices = max(1, ngpus)
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout_rate",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )

        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--pkm_learning_rate", default=1e-2, type=float, help="The initial learning rate for PKM.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        # parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument(
            '--optimizer',
            type=str,
            choices=['adamw', 'adafactor', 'adamp', 'qhadam', 'radam', 'yogi', 'adabound', 'diffgrad', 'acclip', 'sm3'],
            default='adafactor',
        )
