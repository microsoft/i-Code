import hashlib
import json
import re
import sys
from argparse import Namespace
from datetime import date
from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate  # type: ignore

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from benchmarker.config.benchmarker_config import BaseBenchmarkerConfig
from benchmarker.data.model.feature import Feature
from benchmarker.data.utils import FEAT_META, IMG_SIZE_DIVISIBILITY
from benchmarker import MODEL_CLASSES


def calculate_dataset_size(pregenerated_data_path: Path) -> list:
    """
    Calculates size of each training epochs, and number of distinct epoch datasets.
    :param pregenerated_data_path: Path to pregenerated dataset.
    :return: list of numbers of samples per epoch
    """
    samples_per_epoch = []
    if (pregenerated_data_path / "metrics.json").is_file():
        metrics = json.loads((pregenerated_data_path / "metrics.json").read_text())
        return [metrics["num_training_examples"]]
    for i in range(1000):
        epoch_path = pregenerated_data_path / f"epoch_{i}"
        metrics_file = epoch_path / "metrics.json"
        if epoch_path.is_dir() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics["num_training_examples"])
        else:
            if i == 0:
                raise FileNotFoundError("No training data was found!")
            break
    return samples_per_epoch


def join_embedding_types(config: BaseBenchmarkerConfig) -> str:
    embedding_types = [args['embedding_type'] for args in config.context_embeddings]
    if config.page_embeddings_type != "none":
        page_type = [f"page.{config.page_embeddings_type}"]
    else:
        page_type = []
    # limit name of relative embeddings to 3 characters i.e. `horizontal` -> `rel.hor`
    rel_type = ["rel." + args['type'][:3] for args in config.relative_bias_args]
    return "+".join(embedding_types + page_type + rel_type)


#def conf_to_model_name(config: BaseBenchmarkerConfig, training_config: TrainingConfig, num_samples: int) -> str:
def conf_to_model_name(config: BaseBenchmarkerConfig, training_config, num_samples: int) -> str:
    if (training_config.data_dir / "metrics.json").is_file():
        metric_file = training_config.data_dir / "metrics.json"
    else:
        metric_file = training_config.data_dir / "epoch_0" / "metrics.json"
    with open(metric_file, "r") as f:
        num_training_examples = json.loads(f.read())['num_training_examples']

    ce_args = ""
    base_model_dir = str(training_config.model_path).split("/")[-1]
    base_name = (
        base_model_dir
        if len(base_model_dir) < 30
        else re.split("[ ,\-_]", base_model_dir)[0] + "-" + hashlib.md5(base_model_dir.encode()).hexdigest()[:10]
    )
    if (
        len(config.context_embeddings) > 0
        or config.page_embeddings_type != "none"
        or len(config.relative_bias_args) > 0
    ):
        ce_args = hashlib.md5(
            (
                str(config.context_embeddings) + str(config.page_embeddings_args) + str(config.relative_bias_args)
            ).encode()
        ).hexdigest()[:10]
        embedding_description = join_embedding_types(config)
    else:
        embedding_description = 'none'

    if num_samples > 1000000:
        ns = str(int(num_samples / 1000000)) + "m"
    elif num_samples > 1000:
        ns = str(int(num_samples / 1000)) + "k"
    else:
        ns = str(num_samples)
    model_path = ",".join(
        [
            base_name,
            "ce=" + embedding_description,
            "ce-args=" + ce_args,
            "td=" + date.today().strftime("%Y-%m-%d"),
            "ts=" + str(num_training_examples),
            "tt=" + "-".join(config.training_tasks),
            "ns=" + ns,
            "lr=" + str(training_config.learning_rate),
            "ce-w=" + str(config.max_context_weight),
            "d=" + str(config.max_pos_dropout),
        ]
    )
    return model_path


def check_model_type_consistency(model_path: Path, model_type: str) -> None:
    assert model_type in MODEL_CLASSES.keys(), f"Model type {model_type} is not supported"

    contain_mtype_in_name = False
    for mtype in MODEL_CLASSES.keys():
        if mtype + "-" in model_path.stem:
            contain_mtype_in_name = True
    if contain_mtype_in_name:
        assert model_type + "-" in model_path.stem, (
            f"Model picked up with {model_type}, " "but folder name contains different architecture name"
        )
    config = MODEL_CLASSES[model_type]['config'].from_pretrained(str(model_path))
    if hasattr(config, 'model_type'):
        assert (
            config.model_type == model_type
        ), f"Model was trained with different architecture: {config.model_type} != {model_type}"


def load_config(
    model_path: Path, model_type: str = "bert", return_unused_kwargs: bool = False, **kwargs: Any
):
    """
    Saves model params to config.json file. Includes both base Bert configuration, as well as
    additional training and 2D-specific parameters.
    :param model_type: Type of the architecture to use for loading i.e. "bert", "roberta"
    :param args: Input args to training script to be saved to config
    :param model_path: model path, have to contain config.json file
    :param return_unused_kwargs: If True, unused kwargs are returned in an additional dict
    :param kwargs: Dictionary of key, values to update the configuration object after loading.
                Can be used to override selected configuration parameters.
    """
#     check_model_type_consistency(model_path, model_type)

    model_dict = kwargs.copy()
    model_dict['model_type'] = model_type
    model_dict['model_path'] = model_path

    if 'bert_model' in model_dict.keys():
        model_dict['model_path'] = model_dict['bert_model']

    for key, value in model_dict.items():
        if isinstance(value, Path):
            model_dict[key] = str(value)
    return MODEL_CLASSES[model_type]['config2d'].from_pretrained(str(model_path))


def load_tokenizer(
        model_path: Path, model_type: str = "baseline",
        do_lower_case: Optional[bool] = None,
        convert_to_fast_tokenizer: bool = False
):
    """
    Loads BertTokenizer from Bert model directory.

    If `do_lower_case` is explicitly passed, tokenizer will be loaded using that value.
    Otherwise, it is looked up in model's config. If config doesn't contain this parameter,
    BertTokenizer is loaded using `transformers` default behaviour (which is
    checking model identifier for `-cased` or `-uncased` substrings).
    :param model_type: type of the architecture to use for loading i.e. "bert", "roberta"
    :param model_path: model path or identifier. If path, has to contain config.json
    :param do_lower_case: Optional boolean value. Controls BertTokenizer's `do_lower_case`.
    :return: BertTokenizer, RobertaTokenizer or T5Tokenizer
    """
    if do_lower_case is not None:
        tokenizer = MODEL_CLASSES[model_type]['tokenizer'].from_pretrained(
            str(model_path), do_lower_case=do_lower_case
        )
    else:
        config = MODEL_CLASSES[model_type]['config'].from_pretrained(str(model_path))
        if config is None:
            raise FileNotFoundError(f"Provided model or identifier {model_path} is not valid")
        if hasattr(config, "do_lower_case"):
            tokenizer = MODEL_CLASSES[model_type]['tokenizer'].from_pretrained(
                str(model_path), do_lower_case=config.do_lower_case
            )
        else:
            tokenizer = MODEL_CLASSES[model_type]['tokenizer'].from_pretrained(str(model_path))

    if not convert_to_fast_tokenizer or isinstance(tokenizer, PreTrainedTokenizerFast):
        return tokenizer
    return PreTrainedTokenizerFast(__slow_tokenizer=tokenizer)  # Dirty, but worth it


def load_2d_model(
    model_path: Path,
    mode: str,
    state_dict: dict = None,
    args: Namespace = None,
    model_type: str = "bert",
    config: Optional[BaseBenchmarkerConfig] = None,
    **kwargs: Any,
):
    """

    :param config: config to use to load models, if None config will be picked up from model_path
    :param model_path: model path or identifier. If path, has to contain config.json and weights
    :param mode: mode defining the class which will be used to load the model
    :param state_dict: Optional dict with weights to use to initialize the model
    :param args: additional Namespace of arguments which will be added to config of the model
    :param model_type: type of the architecture to use for loading i.e. "bert", "roberta"
    :param kwargs: additional arguments which will be added to config of the model
    :return:
    """
    if config is None:
        config = load_config(
            model_path, model_type=model_type, args=args, return_unused_kwargs=False, **kwargs  # type: ignore
        )
    if mode == 'pretraining':
        model = MODEL_CLASSES[model_type]['pretraining'].from_pretrained(
            str(model_path), config=config, state_dict=state_dict
        )
    elif mode == 'embedding':
        model = MODEL_CLASSES[model_type]['embedding'].from_pretrained(
            str(model_path), config=config, state_dict=state_dict)
    elif mode == 'token_classification':
        model = MODEL_CLASSES[model_type]['token_classification'].from_pretrained(
            str(model_path), config=config, state_dict=state_dict)
    else:
        raise NotImplementedError(
            "There is no model mode implemented for "
            + f"given string {mode}"
        )

    return model  # type: ignore


def features_collate(batch: Sequence[Feature]) -> Dict[str, Any]:
    dict_batch = {}
    token_label_ids_batch = [feat.token_label_ids for feat in batch]
    dict_batch['token_label_ids'] = cast(Dict[str, Any], default_collate(token_label_ids_batch))
    lm_label_ids_batch = [feat.lm_label_ids for feat in batch]
    dict_batch['lm_label_ids'] = cast(Dict[str, Any], default_collate(lm_label_ids_batch))
    input_masks_batch = [feat.input_masks for feat in batch]
    dict_batch['input_masks'] = cast(Dict[str, Any], default_collate(input_masks_batch))
    input_ids_batch = [feat.input_ids for feat in batch]
    dict_batch['input_ids'] = cast(Dict[str, Any], default_collate(input_ids_batch))
    seg_data_batch = [feat.seg_data for feat in batch]
    dict_batch['seg_data'] = dict_collate(seg_data_batch)
    return dict_batch


def dict_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(batch[0], dict):
        dict_batch = {}
        for k in batch[0].keys():
            if k == "img_lst":
                # assuming 3 channels, temporary and only for DALLE
                dict_batch[k] = merge_images_into_tensor([el[k].permute(2,0,1) for el in batch], 
                                                         IMG_SIZE_DIVISIBILITY)
            else:
                dict_batch[k] = dict_collate([el[k] for el in batch])
        return prepare_batch_dict(dict_batch)
    elif batch[0] is None:
        return None  # type: ignore
    else:
        return cast(Dict[str, Any], default_collate(batch))


def merge_images_into_tensor(
        tensors: list, size_divisibility: int = 64, pad_value: float = 255.
    ) -> "torch.Tensor":
    """
    Copied from detectron2
    Args:
        tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
            (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
            to the same shape with `pad_value`.
        size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
            the common height and width is divisible by `size_divisibility`.
            This depends on the model and many models need a divisibility of 32.
        pad_value (float): value to pad

    Returns:
        an `ImageList`.
    """
    assert len(tensors) > 0
    assert isinstance(tensors, (tuple, list))
    for t in tensors:
        assert isinstance(t, torch.Tensor), type(t)
        assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

    image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
    image_sizes_tensor = [_as_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values

    if size_divisibility > 1:
        stride = size_divisibility
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size + (stride - 1)) // stride * stride

    # max_size can be a tensor in tracing mode, therefore convert to list
    batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)
    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs.contiguous()


def prepare_batch_dict(batch: Dict[str, Any], device: Union[str, torch.device, None] = None) -> Dict[str, Any]:
    for skey, seg in batch.items():
        if isinstance(seg, dict):
            batch[skey] = prepare_batch_dict(seg, device)
        elif isinstance(seg, torch.Tensor):
            batch[skey] = seg.to(dtype=FEAT_META[skey]['train_dtype'], device=device)
    return batch


def _as_tensor(x: Tuple[int, int]) -> torch.Tensor:
    """
    An equivalent of `torch.as_tensor`, but works under tracing if input
    is a list of tensor. `torch.as_tensor` will record a constant in tracing,
    but this function will use `torch.stack` instead.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x)
    if isinstance(x, (list, tuple)) and all([isinstance(t, torch.Tensor) for t in x]):
        return torch.stack(x)
    return torch.as_tensor(x)


def dict_collate_trim_l5(batch: Sequence[Dict[str, Any]],
                         input_len=None, target_len=None, divisibility=32) -> Dict[str, Any]:
    if input_len is None:
        input_len = max([s["attention_mask"].sum() for s in batch])
        input_len = (input_len + (divisibility - 1)) // divisibility * divisibility
    if target_len is None:
        target_len = max([np.count_nonzero(s["labels"]) for s in batch]) + 1
        target_len = (target_len + (divisibility - 1)) // divisibility * divisibility
    if isinstance(batch[0], dict):
        dict_batch = {}
        for k in batch[0].keys():
            dict_batch[k] = dict_collate_trim_l5([el[k] for el in batch], input_len, target_len)
        return prepare_batch_dict_trim_l5(dict_batch, input_len=input_len, target_len=target_len)
    elif batch[0] is None:
        return None  # type: ignore
    else:
        return cast(Dict[str, Any], default_collate(batch))


def prepare_batch_dict_trim_l5(batch: Dict[str, Any], device: Union[str, torch.device, None] = None,
                               input_len=None, target_len=None) -> Dict[str, Any]:
    assert input_len is not None and target_len is not None
    for skey, seg in batch.items():
        if isinstance(seg, dict):
            batch[skey] = prepare_batch_dict_trim_l5(seg, device, input_len, target_len)
        elif isinstance(seg, torch.Tensor):
            batch[skey] = seg.to(dtype=FEAT_META[skey]['train_dtype'], device=device)
            if skey in ("attention_mask", "bboxes", "input_ids", "ranges", "masks", "token_map"):
                batch[skey] = batch[skey][:, :input_len].contiguous()
            elif skey == "labels":
                batch[skey] = batch[skey][:, :target_len].contiguous()

    return batch


def save_checkpoint(model, tokenizer, epoch, output_dir):
    """
    Save model checkpoint.

    The checkpoint contains of model, config and tokenizer, to assure that it can be used
    just as normal model.
    :param model: Model to save.
    :param epoch: Number of current epoch (used for naming checkpoint), use None for saving final model
    :param iteration: Number of current iteration in current epoch (used for naming checkpoint)
    :param output_dir: Directory to save file to.
    """
    if epoch is None:
        checkpoint_dir = output_dir
    else:
        checkpoint_dir = output_dir / f"checkpoint_e{epoch}"

    # since we allow output directory to already exist in training script,
    # it would be surprising if it didn't work the same for subdirectories

    checkpoint_dir.mkdir(exist_ok=True)
    to_save = model.module if hasattr(model, "module") else model
    tokenizer.save_pretrained(checkpoint_dir)
    to_save.save_pretrained(checkpoint_dir)


def get_total_samples(max_epochs: int, max_train_samples: int, pregenerated_data: Path):
    """
    Calculates number of training samples when either number of epochs or number of
    samples is specified.
    :param max_epochs: number of epochs. Default lightning max_epochs equals 1000.
    :param max_train_samples: number of training samples. sys,maxsize means "don't care".
    :param pregenerated_data: path to training dataset.
    """
    assert (max_epochs != 1000) ^ (max_train_samples is not sys.maxsize)

    if max_epochs != 1000:
        total_samples = 0
        dataset_size = calculate_dataset_size(pregenerated_data)
        for i in range(max_epochs):
            total_samples += dataset_size[i % len(dataset_size)]
    else:
        total_samples = max_train_samples
    return total_samples


def get_total_steps(
    max_epochs: int,
    max_train_samples: int,
    pregenerated_data: Path,
    single_gpu_batch_size: int,
    accumulate_grad_batches: int,
):
    """
    Calcualtes number of training steps (optimizer steps) when either number of epochs or
    number of samples is specified.from
    :param max_epochs: number of epochs. Default lightning max_epochs equals 1000.
    :param max_train_samples: number of training samples. sys,maxsize means "don't care".
    :param pregenerated_data: path to training dataset.
    :param single_gpu_batch_size: number of samples loaded on GPU at once.
    :param accumulate_grad_batches: after how many batches on GPU update optimizer
            (if there are multible GPUs, batch_size * accumulate_grad_batches != effective_batch_size).
    """
    assert (max_epochs != 1000) ^ (max_train_samples is not sys.maxsize)

    if max_epochs != 1000:
        total_steps = 0
        dataset_size = calculate_dataset_size(pregenerated_data)

        for i in range(max_epochs):
            epoch_samples = dataset_size[i % len(dataset_size)]
            # Lightning tends to cut off few last samples (idk why),
            # +1 enforces last epoch ending (and calling validation).
            epoch_steps = int(ceil(epoch_samples / single_gpu_batch_size / accumulate_grad_batches)) + 1
            total_steps += epoch_steps
    else:
        total_steps = int(ceil(max_train_samples / single_gpu_batch_size / accumulate_grad_batches))

    assert total_steps != 0, "max_train_samples too low."

    return total_steps


def calculate_accumulate_grad_batches(gpus_num: int, eff_batch_size: int, single_gpu_batch_size: int) -> int:
    if eff_batch_size % (gpus_num * single_gpu_batch_size) != 0:
        raise ValueError("Effective batch size should be a divisible by single_gpu_batch_size * gpus_num.")

    return int(eff_batch_size / (gpus_num * single_gpu_batch_size))

