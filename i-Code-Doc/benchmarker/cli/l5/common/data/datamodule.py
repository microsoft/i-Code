import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
from catalyst.data import DistributedSamplerWrapper
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataset import ConcatDataset, Dataset

from benchmarker.utils.pregenerated import PregeneratedCustomDataset
from benchmarker.utils.training import dict_collate, dict_collate_trim_l5

logger = logging.getLogger(__name__)


class L5DataModule(pl.LightningDataModule):

    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.segment_levels = tuple(self.hparams.segment_levels)
        self.additional_data_fields = tuple(self.hparams.additional_data_fields)
        self.input_len = self.hparams.max_source_length
        self.trim_batches = self.hparams.trim_batches
        self.collate = dict_collate_trim_l5 if self.trim_batches else dict_collate
        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        self.data_dir = {
            "train": os.path.join(self.hparams.data_dir[0], self.hparams.train_data_dir[0]),
            "val": os.path.join(self.hparams.data_dir[0], self.hparams.val_data_dir[0]),
            "test": os.path.join(self.hparams.data_dir[0], self.hparams.test_data_dir[0]),
        }
        self.datasets = {}
        self.datasets_weights = self.hparams.datasets_weights
        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size
        self.num_workers = self.hparams.num_workers
        self.img_conf = self.hparams.img_conf
        
        self.im_dir = os.path.join(self.hparams.data_dir[0], self.hparams.im_dir[0])

    def setup(self, stage):
        [self._setup_datasets(directory, split) for split, directory in self.data_dir.items() if directory]

    def _setup_datasets(self, directory: Any, split: str):
        if isinstance(directory, list):
            datasets_list = [self._load_dataset(d, split) for d in directory]
            self.datasets[split] = ConcatDataset(datasets_list) if len(directory) > 1 else datasets_list[0]
        else:
            self.datasets[split] = self._load_dataset(directory, split)

    def _load_dataset(self, directory: Path, split: str) -> Dataset:
        additional_data_fields = ("doc_id",) if split == "train" else self.additional_data_fields
        dataset = PregeneratedCustomDataset.load_from_memmap(
            im_dir=self.im_dir,
            path=Path(directory),
            segment_levels=self.segment_levels,
            additional_memmap_files=("lm_label_ids",) + additional_data_fields,
            img_conf=self.img_conf,
        )
        self.rename_keys(dataset.data)
        self.resize_data(dataset.data, split)
        return dataset

    @staticmethod
    def rename_keys(data_dict):
        # TODO: remove it once names in the data_converter are aligned with names in the transformers
        data_dict["labels"] = data_dict.pop("lm_label_ids")
        data_dict["attention_mask"] = data_dict.pop("input_masks")

    def resize_data(self, data, split):
        target_len = self.target_lens[split]
        input_len = self.input_len if split == "train" else -1
        if input_len > 0:
            data["input_ids"] = self.resize_mmap(data["input_ids"], input_len)
            data["attention_mask"] = self.resize_mmap(data["attention_mask"], input_len)
            data["seg_data"]["tokens"]["bboxes"] = self.resize_mmap(data["seg_data"]["tokens"]["bboxes"], input_len)
        if target_len > 0:
            data["labels"] = self.resize_mmap(data["labels"], target_len)

    @staticmethod
    def resize_mmap(mmap, new_size):
        mmap_size = mmap.shape[1]
        if mmap_size < new_size:
            raise ValueError("Memmap seq size is not sufficient for required length")
        return mmap[:, :new_size]

    def train_dataloader(self):
        if "train" in self.datasets:
            sampler = self._get_sampler("train")
            shuffle = sampler is None
            return DataLoader(
                self.datasets["train"],
                batch_size=self.train_batch_size,
                collate_fn=self.collate,
                drop_last=False,
                num_workers=self.num_workers,
                shuffle=shuffle,
                sampler=sampler,
                pin_memory=True,
            )

    def _get_sampler(self, split) -> Optional[Sampler]:
        # for some resaon PL is not replacing dataloader's samplers if train_dataloader
        # have custom sampler, for such cases each dataloader should be converted manually do ddp
        if isinstance(self.datasets["train"], ConcatDataset):
            dataset = self.datasets[split]
            if split == "train":
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            if self._gpu_count() > 1:
                sampler = DistributedSamplerWrapper(sampler)
            return sampler
        return None

    def _gpu_count(self):
        if isinstance(self.hparams.gpus, int):
            return self.hparams.gpus
        elif isinstance(self.hparams.gpus, list):
            return len(self.hparams.gpus)
        elif isinstance(self.hparams.gpus, str):
            return len(re.findall(r'\d+', self.hparams.gpus))
        return 0

    def val_dataloader(self):
        if "val" in self.datasets:
            sampler = self._get_sampler("val")
            return DataLoader(
                self.datasets["val"],
                batch_size=self.eval_batch_size,
                collate_fn=self.collate,
                drop_last=False,
                num_workers=self.num_workers,
                shuffle=False,
                sampler=sampler,
                pin_memory=True,
            )

    def test_dataloader(self):
        if "test" in self.datasets:
            sampler = self._get_sampler("test")
            return DataLoader(
                self.datasets["test"],
                batch_size=self.eval_batch_size,
                collate_fn=self.collate,
                drop_last=False,
                num_workers=self.num_workers,
                shuffle=False,
                sampler=sampler,
                pin_memory=True,
            )

    @staticmethod
    def add_data_specific_args(parser):
        parser.add_argument(
            "--data_dir", default=None, type=Path, nargs='+', help="The sequence of input data directories."
        )
        parser.add_argument(
            "--datasets_weights",
            default=None,
            type=float,
            nargs='+',
            help="A sequence of weights (one weight per each dataset), not necessary summing up to one. "
            "The weights decide how often to sample from each train dataset.",
        )
        parser.add_argument("--train_data_dir", default=None, type=Path, nargs="+", help="The input data dir.")
        parser.add_argument("--val_data_dir", default=None, type=Path, nargs="+", help="The input data dir.")
        parser.add_argument("--test_data_dir", default=None, type=Path, nargs="+", help="The input data dir.")
        parser.add_argument("--im_dir", default=None, type=Path, nargs="+", help="The input img data dir.")
        parser.add_argument(
            "--max_source_length",
            default=-1,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. This applies only for"
            "train set",
        )
        parser.add_argument(
            "--max_target_length",
            default=-1,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=-1,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=-1,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)
        parser.add_argument(
            "--segment_levels",
            nargs='+',
            type=str,
            default=["tokens", "pages"],
            required=False,
            help="2D information which will be loaded by Dataloaders",
        )
        parser.add_argument(
            "--additional_data_fields",
            nargs='+',
            type=str,
            default=["doc_id", "label_name"],
            required=False,
            help="additional fields which will be loaded by test&val Dataloaders",
        )
        parser.add_argument(
            "--trim_batches",
            action="store_true",
            default=False,
            help="whether to trim batches to longest element in batch to save computing time",
        )
        parser.add_argument(
            "--img_conf",
            type=json.loads,
            default='{"width":768,"max_height":10240,"channels":1,'
                    '"imtok_per_width":3,"imtok_id":32000}',
            help="Options defining how to prepare images by dataloader",
        )

        return parser
