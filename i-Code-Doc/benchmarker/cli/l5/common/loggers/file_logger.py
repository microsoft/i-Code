# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import csv
import io
import os
from pathlib import Path
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only


class FileExperimentWriter(object):
    """
    Experiment writer for FileLogger.

    Currently supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_HPARAMS_FILE = 'hparams.yaml'
    NAME_METRICS_FILE = 'metrics.csv'

    def __init__(self, log_dir: str) -> None:
        self.hparams = {}
        self.metrics = []

        self.log_dir = log_dir
#         if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
#             rank_zero_warn(
#                 f"Experiment logs directory {self.log_dir} exists and is not empty."
#                 " Previous log files in this directory will be deleted when the new ones are saved!"
#             )
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams"""
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics"""

        def _handle_value(value):
            if isinstance(value, int):
                return value
            if isinstance(value, torch.Tensor):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.item()
            if isinstance(value, np.float64):
                return float(value)
            return float(value)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}

        if any(map(lambda x: x.startswith('val_'), metrics_dict.keys())):
            if step is None:
                return
            metric_file = Path(self.log_dir) / 'val_results' / f'{step:05d}.txt'
            metric_file.parent.mkdir(parents=True, exist_ok=True)
            save_hparams_to_yaml(metric_file, metrics)

        elif any(map(lambda x: x.startswith('test_'), metrics_dict.keys())):
            metric_file = Path(self.log_dir) / 'test_results.txt'
            metric_file.parent.mkdir(parents=True, exist_ok=True)
            save_hparams_to_yaml(metric_file, metrics)
        else:
            return

        if step is None:
            step = len(self.metrics)

        metrics['step'] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded hparams and metrics into files"""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        for k in self.hparams:
            if isinstance(self.hparams[k], Path):
                self.hparams[k] = str(self.hparams[k])
            if isinstance(self.hparams[k], list):
                self.hparams[k] = list(map(str, self.hparams[k]))

        save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.metrics:
            return

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with io.open(self.metrics_file_path, 'w', newline='') as f:
            self.writer = csv.DictWriter(f, fieldnames=metrics_keys)
            self.writer.writeheader()
            self.writer.writerows(self.metrics)


class FileLogger(LightningLoggerBase):
    r"""
    Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
    """

    LOGGER_JOIN_CHAR = '-'

    def __init__(
        self, save_dir: str, version: Optional[Union[int, str]] = None, prefix: str = '',
    ):
        super().__init__()
        self._log_dir = save_dir
        self._version = version
        self._prefix = prefix
        self._experiment = None

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> FileExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.log_dir, exist_ok=True)
        self._experiment = FileExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
#         params = self._convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # metrics = self._add_prefix(metrics)
        self.experiment.log_metrics(metrics, step)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        return 'FileLogger'

    @property
    def version(self) -> int:
        return 0
