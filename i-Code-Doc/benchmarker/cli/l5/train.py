#!/usr/bin/env python

import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type import DDPPlugin

from benchmarker.cli.l5.common.callbacks import (
    CustomProgressBar,
    SavePredictionCallback,
    SaveTransformerCheckpoint,
    get_checkpoint_callback,
    get_early_stopping_callback,
)
from benchmarker.cli.l5.common.data.datamodule import L5DataModule
from benchmarker.cli.l5.common.loggers.better_mlflow_logger import BetterMlFlowLogger
from benchmarker.cli.l5.common.loggers.file_logger import FileLogger
from benchmarker.cli.l5.common.metrics import Accuracy
from benchmarker.cli.l5.common.pl_modules.l5_generation_module import L5GenerationModule
from benchmarker.cli.l5.common.utils import check_output_dir
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 
# torch.set_num_threads(1)

def main(args: Namespace, model: Optional[L5GenerationModule] = None, datamodule=None,
         callbacks=None, process_position=0) -> Tuple[L5GenerationModule, pl.Trainer]:

    Path(args.output_dir).mkdir(exist_ok=True)
    check_output_dir(args, expected_items=3)

    if model is None:
        val_metrics = {}#setup_val_metrics(args.val_metric)
        model: L5GenerationModule = L5GenerationModule(args, val_metrics=val_metrics)

    if datamodule is None:
        datamodule = L5DataModule(args)

    if callbacks is None:
        callbacks = setup_callbacks(args, process_position=process_position)

    loggers = setup_loggers(args)

    trainer: pl.Trainer = get_generic_trainer(model, datamodule, args, callbacks, loggers=loggers)

    if args.do_predict:
        # noinspection PyTypeChecker
        predict(args, model, trainer, datamodule)

    return model, trainer


def setup_val_metrics(metric_names: List[str]):
    valid_metrics = {}
    for metric_name in metric_names:
        if metric_name == 'accuracy':
            valid_metrics['accuracy'] = Accuracy(compute_on_step=False)
    return valid_metrics


def setup_callbacks(args: Namespace, process_position=0) -> List[pl.Callback]:
    callbacks = []

    if args.early_stopping_patience >= 0:
        es_metric = args.val_metric[0] if args.val_metric else 'loss'
        es_callback = get_early_stopping_callback(es_metric, args.early_stopping_patience)
        callbacks.append(es_callback)

    lower_is_better = args.val_metric[0] == "loss"
    callbacks.append(get_checkpoint_callback(args.output_dir, args.val_metric[0], args.save_top_k, lower_is_better))
    callbacks.append(SavePredictionCallback())
    callbacks.append(SaveTransformerCheckpoint(save_path=Path(args.output_dir) / 'best_tfmr'))
    callbacks.append(CustomProgressBar(process_position=process_position))
    return callbacks

# FIXME
def setup_loggers(args: Namespace) -> List[pl.loggers.LightningLoggerBase]:
    loggers = [FileLogger(args.output_dir)]
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        pass
    elif args.logger_name == "mlflow":
        loggers.append(BetterMlFlowLogger(args.mlflow_experiment, args.mlflow_uri, args.mlflow_tags))
    else:
        raise NotImplementedError(f"{args.logger_name} logger is not implemented")
    return loggers


def get_generic_trainer(model: L5GenerationModule, datamodule: L5DataModule, args: Namespace,
                        callbacks: List[pl.Callback], loggers: List[pl.loggers.LightningLoggerBase]) -> pl.Trainer:
    # copy of the transformers example code with the exception of passing datamodule to fit
    pl.seed_everything(args.seed)

    # init model
    
    Path(model.hparams.output_dir).mkdir(exist_ok=True, parents=True)

    if args.restore_training and (Path(args.output_dir) / 'last.ckpt').exists():
        resume_from_checkpoint = str(Path(args.output_dir) / 'last.ckpt')
    elif hasattr(args, 'resume_from_checkpoint'):
        resume_from_checkpoint = args.resume_from_checkpoint
    else:
        resume_from_checkpoint = None

    train_params = {'accumulate_grad_batches': args.accumulate_grad_batches,
        'replace_sampler_ddp': len(args.data_dir) == 1, 'plugins': None}

    if args.num_nodes > 1:
        # See https://applica.atlassian.net/browse/AA-751
        # or https://pytorch-lightning.readthedocs.io/en/1.3.8/benchmarking/performance.html for details
        train_params['plugins'] = [DDPPlugin(find_unused_parameters=True)]

    train_params['accelerator'] = 'gpu'
    trainer = pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=callbacks,
        logger=loggers,
        progress_bar_refresh_rate=0,  # Disables default progress bar
        resume_from_checkpoint=resume_from_checkpoint,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model, datamodule=datamodule)

    return trainer


def predict(args: Namespace, model: L5GenerationModule, trainer: pl.Trainer, datamodule: L5DataModule):
    model.hparams.test_checkpoint = ""
    trainer.logger.log_hyperparams(model.hparams)
    trainer.test(model, datamodule=datamodule, verbose=False)


def get_train_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")    # FIXME: wziąc do readme
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--val_metric",
        type=str,
        default=['loss'],
        nargs="+",
        required=False,
        choices=["accuracy", "loss", "f1", 'anls'],
    )

    return parser


if __name__ == "__main__":
    parser_ = get_train_parser()
    parser_ = pl.Trainer.add_argparse_args(parser_)
    parser_ = L5GenerationModule.add_model_specific_args(parser_, os.getcwd())
    parser_ = L5DataModule.add_data_specific_args(parser_)

    main(parser_.parse_args())
