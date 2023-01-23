import logging
from pathlib import Path

import numpy as np
import jsonlines
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class SavePredictionCallback(pl.Callback):
    """
    Copy of Seq2SeqLoggingCallback from transformers, which include few modifications related to generations saving
    and minor fixes
    """

    @property
    def is_rank_zero(self):
        return rank_zero_only.rank == 0

    def save_predictions(self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str) -> None:
#         metrics = trainer.callback_metrics
        metrics = pl_module.val_outs
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == 'test':
            generations_file = od / 'test_generations.txt'
        else:
            generations_file = od / f'{type_path}_generations/{trainer.global_step:05d}.txt'
            generations_file.parent.mkdir(exist_ok=True)

        if 'generation_results' in metrics:
            generations = metrics['generation_results']

            if self.is_rank_zero:
                for i in range(min(len(generations), 10)):
                    logger.info(f'pred:\t {generations[i]["preds"]}')
                    logger.info(f'target:\t {generations[i]["target"]}')

            self._save_generations(generations, generations_file)

    @staticmethod
    def _save_generations(generations, path):
        with open(path, "a+") as f:
            with jsonlines.Writer(f) as writer:
                writer.write_all(generations)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.save_predictions(trainer, pl_module, "test")

    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        self.save_predictions(trainer, pl_module, "val")
