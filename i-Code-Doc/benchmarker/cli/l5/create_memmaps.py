#!/usr/bin/env python
import logging
from pathlib import Path
from typing import Iterator

import fire

from benchmarker.cli.l5.common.utils import save_t5_kleister_cache
from benchmarker.data.model.feature import Feature
from benchmarker.data.reader import Corpus, qa_strategies
from benchmarker.data.slicer import LongPageStrategy
from benchmarker.data.t5 import T5DownstreamDataConverter
from benchmarker.utils.training import load_tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def list_wrapper(features_iterator: Iterator[Feature], limit: int = -1):
    for idx, feature in enumerate(features_iterator):
        if idx == limit:
            break
        if feature is None:
            continue
        yield [feature]


def generate_memmaps(
    dataset_path_or_name: str,
    model_path: str,
    memmap_path: str,
    max_encoder_length: int = 512,
    model_type: str = 't5',
    unescape_prefix: bool = False,
    unescape_values: bool = True,
    use_prefix: bool = True,
    prefix_separator: str = ':',
    values_separator: str = '|',
    single_property: bool = True,
    use_none_answers: bool = False,
    use_fast_tokenizer: bool = False,
    limit: int = -1,
    case_augmentation: bool = False,
    segment_levels: tuple = ('tokens', 'lines'),
    long_page_strategy: str = 'FIRST_PART',
    ocr_engine: str = 'tesseract',
    lowercase_expected: bool = False,
    lowercase_input: bool = False,
    train_strategy: str = 'first_item',
    dev_strategy: str = 'concat',
    test_strategy: str = 'concat',
    augment_tokens_from_file: str = '',
    img_matrix_order: int = 0,
    processes=1,
    imap_chunksize=100,
    skip_text_tokens=False,
):
    r"""
    Generate memmaps for given dataset.

    Recommended tensorflow datasets for docvqa pretraining (only answerable):
        --dataset_path_or_name 'squad/v1.1' --tfds_split 'train+validation'
        --dataset_path_or_name coqa --tfds_split 'train+test'
        --dataset_path_or_name tydi_qa --tfds_split 'train+validation-en' --tfds_match_regex '{"id":"^english"}'
        --dataset_path_or_name 'quac' --tfds_split 'train+validation' --tfds_match_regex '{("orig_answer", "text"): "^(?!CANNOTANSWER)"}'
        --dataset_path_or_name 'race/high' --tfds_split 'train+dev+test'
        --dataset_path_or_name 'race/middle' --tfds_split 'train+dev+test'
        --dataset_path_or_name drop --tfds_split 'train+dev'
        --dataset_path_or_name qasc --tfds_split 'train+validation'

    Args:
        dataset_path_or_name: kleister challenge root path or tfds dataset name
        model_path: Path to pre-trained tokenizer
        memmap_path: Output path of the memmaps
        max_encoder_length: maximum sequence length for encoder
        model_type: type of enc-dec model to use
        unescape_prefix: whether unescape prefix (e.g., replace _ with spaces)
        unescape_values: whether unescape values (e.g., replace _ with spaces)
        use_prefix: if set to True, input document will be prefixed with property name
        prefix_separator: string to place between property name and input text
        values_separator: string to place between values in a case of multi-value properties
        single_property: whether assume single-property inference (see Multi-property extraction paper)
        use_none_answers: whether to use 'None' as answer when no answer is present
        use_fast_tokenizer: wheter to convert T5Tokenizer to PreTrainedTokenizerFast (x100 speed up)
        limit: maximum number of features to generate
        case_augmentation: whether to case-augment training documents
        segment_levels: Define which visual segments levels to load
        ocr_engine: name of OCR engine. Used only in json datasets. Choices are dataset-dependant, usually some of ('tesseract', 'ms32', 'textract', 'external')
        lowercase_expected: whether to lowercase expected system output
        lowercase_input: lowercase input document (including prefix)
        train_strategy: Strategy for choosing expected items from trainset. Choices: ('concat', 'first_item', 'all_items', 'shortest', 'longest')
        dev_strategy: Strategy for choosing expected items from devset. Choices: ('concat', 'first_item', 'all_items', 'shortest', 'longest')
        test_strategy: Strategy for choosing expected items from testset. Choices: ('concat', 'first_item', 'all_items', 'shortest', 'longest')
        img_matrix_order: Order of img bbox matrix, it's a square root of image tokens count (e.g. img_matrix_order==4 means having 16 image tokens)
        processes: number of threads to use for preparing data
        imap_chunksize: chop the docs iterable into a number of chunks which will be submited to the process pool as separate tasks
        skip_text_tokens: whether to not use text tokens as an input. Useful for latter use of image tokens

    """
    model_path, memmap_path = Path(model_path), Path(memmap_path)

    corpus = Corpus(
        unescape_prefix=unescape_prefix,
        unescape_values=unescape_values,
        use_prefix=use_prefix,
        prefix_separator=prefix_separator,
        values_separator=values_separator,
        single_property=single_property,
        use_none_answers=use_none_answers,
        case_augmentation=case_augmentation,
        lowercase_expected=lowercase_expected,
        lowercase_input=lowercase_input,
        train_strategy=getattr(qa_strategies, train_strategy),
        dev_strategy=getattr(qa_strategies, dev_strategy),
        test_strategy=getattr(qa_strategies, test_strategy),
        augment_tokens_from_file=augment_tokens_from_file,
    )

    corpus.read_benchmark_challenge(
        directory=Path(dataset_path_or_name),
        ocr=ocr_engine,
        segment_levels=segment_levels,
    )

    tokenizer = load_tokenizer(model_path, model_type=model_type, convert_to_fast_tokenizer=use_fast_tokenizer)
    data_converter = T5DownstreamDataConverter(
        tokenizer,
        segment_levels=segment_levels,
        max_seq_length=max_encoder_length,
        long_page_strategy=LongPageStrategy(long_page_strategy),
        img_matrix_order=img_matrix_order,
        processes=processes,
        imap_chunksize=imap_chunksize,
        skip_text_tokens=skip_text_tokens,
    )
    for set_name in ('train', 'dev', 'test'):
        subset = getattr(corpus, set_name)
        if not subset:
            continue
        train_features = data_converter.generate_features(subset)
        train_features = list_wrapper(train_features, limit)
        save_t5_kleister_cache(memmap_path / set_name, tokenizer, train_features, max_encoder_length, segment_levels)


if __name__ == '__main__':
    fire.Fire(generate_memmaps)
