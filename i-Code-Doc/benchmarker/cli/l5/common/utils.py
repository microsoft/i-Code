import csv
import itertools
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import pandas as pd
from torch import nn

from benchmarker.utils.pregenerated import PregeneratedCustomDataset

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


def save_t5_kleister_cache(path, tokenizer, features, max_seq_length, segment_levels=("tokens",)):
    _ = PregeneratedCustomDataset(
        path=path,
        tokenizer=tokenizer,
        input_data=features,  # type: ignore
        mode='w+',
        max_seq_length=max_seq_length,
        additional_memmap_files=("lm_label_ids", "doc_id", "label_name"),
        segment_levels=segment_levels,
    )


def add_tag_to_path(path, tag):
    path = str(path)
    assert path[-4:] == ".tsv"
    tag = tag.replace(".", "_")
    path_out = path[:-4] + f",{tag}.tsv"
    return path_out


def get_in_file(path: Path) -> List[Tuple[str, str]]:
    df = pd.read_csv(str(path / "in.tsv"), sep=r"\t", header=None, engine="python")
    doc_names = list(df[0].values)
    if len(df.columns) >= 3:
        doc_texts = list(df[2].values)
        doc_texts = [t if isinstance(t, str) else "" for t in doc_texts]
    else:
        doc_texts = ["" for _ in doc_names]
    return list(zip(doc_names, doc_texts))


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
