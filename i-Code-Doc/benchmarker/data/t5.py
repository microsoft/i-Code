import logging
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from benchmarker.data.data_converter import DataConverter
from benchmarker.data.model.feature import Feature
from benchmarker.data.model.span import Span
from benchmarker.data.reader.common import DataInstance
from benchmarker.data.slicer import LongPageStrategy
from benchmarker.data.utils import single_line_spans

logger = logging.Logger(__name__)


def noise_span_to_unique_sentinel(tokens, noise_mask, tokenizer):
    """Partially copied from original text-to-text-transfer-transformer repo.
    Replace each run of consecutive noise tokens with a different sentinel.

    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.

    We want to generate training examples like
    "We hold X to be Y that Z" -> "X these truths Y self evident Z that"

    Sentinels assigned in decreasing order within the sequence starting at
    vocabulary.size - 1.  That is, we appropriate the last tokens in the
    vocabulary for additional use as sentinels.

    :param tokens: a 1d integer Tensor
    :param noise_mask: a boolean Tensor with the same shape as tokens
    :param tokenizer: a t5 tokenizer
    :return: a Tensor with the same shape and dtype as tokens
    """
    vocab_size = tokenizer.vocab_size
    prev_token_is_noise = np.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = np.logical_and(noise_mask, np.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)

    sentinel_ids = vocab_size - np.cumsum(first_noise_tokens)
    sentinel_tokens = tokenizer.convert_ids_to_tokens(sentinel_ids)

    tokens = np.where(first_noise_tokens, sentinel_tokens, tokens)
    return tokens[np.logical_not(subsequent_noise_tokens)]


def convert_ranges_to_noise_mask(noise_spans_ranges, length):
    single_line_span_starts = noise_spans_ranges.flatten()
    span_start_indicator = np.zeros(length)
    span_start_indicator[single_line_span_starts[:-1]] = 1
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise[:length]


def random_spans_noise_mask(length, seg_data, noise_density=0.2, mean_noise_span_length=3.5):
    """Partially coppied from text-to-text-transfer-transformer git repo
    :param length: number of tokens
    :param seg_data: dictionary with segment/visual data
    :param noise_density: what fraction of the tokens to select as noise
    :param mean_noise_span_length: average length of noise span, in the end actual
        average span length will be lover due to span truncation to the one-line span

    """

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = np.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        :param num_items: an integer scalar > 0
        :param num_segments: an integer scalar in [1, num_items]
        :return: a Tensor with shape [num_segments] containing positive integers that add
          up to num_items
        """
        first_in_segment = np.pad(
            np.random.permutation((np.arange(num_items - 1) < num_segments - 1).astype(np.int)), [[1, 0]]
        )
        segment_id = np.cumsum(first_in_segment)
        # segment_length = np.segment_sum(np.ones_like(segment_id), segment_id)
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)
    noise_spans_ranges = span_starts.reshape(-1, 2)
    noise_spans_ranges_single_line = single_line_spans(noise_spans_ranges, seg_data)
    is_noise = convert_ranges_to_noise_mask(noise_spans_ranges_single_line, length)
    return is_noise, noise_spans_ranges_single_line


def _recompute_seg_data_for_sentinels(seg_data, noise_mask, noise_spans_ranges):
    """
    modify tokens bboxes and segment ranges to match modified input
    example:
        original:
        input: Tom went to the store to buy sth
        lines['ranges']: [0,7]
        tokens['bboxes']: [bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7]
        t5:
        input: Tom went <sentinel_1> to buy sth
        lines['ranges']: [0,5]
        tokens['bboxes']: [bb0, bb1, bb2-bb4, bb5, bb6, bb7]

    :param seg_data: original seg_data
    :param noise_mask: mask
    :return: modified seg_data
    """

    tok_bboxes = seg_data["tokens"]["bboxes"]
    is_token_mask = np.ones(len(tok_bboxes), dtype=np.bool)
    for span in noise_spans_ranges:
        tok_bboxes[span[0] : span[1]] = np.concatenate(
            [tok_bboxes[span[0] : span[1], 0:2].min(axis=0), tok_bboxes[span[0] : span[1], 2:4].max(axis=0)]
        )
        is_token_mask[span[0] + 1 : span[1]] = False

    seg_data["tokens"]["bboxes"] = tok_bboxes[is_token_mask]
    seg_data["tokens"]["org_bboxes"] = seg_data["tokens"]["org_bboxes"][is_token_mask]

    tok_range_list = []
    prev_x1 = 0
    for x0, x1 in noise_spans_ranges:
        for i in range(prev_x1, x0):
            tok_range_list.append([i, i + 1])
        tok_range_list.append([x0, x1])
        prev_x1 = x1
    for i in range(prev_x1, len(noise_mask)):
        tok_range_list.append([i, i + 1])

    tok_range = np.array(tok_range_list)

    for skey, seg in seg_data.items():
        if skey not in ("tokens", "images", "lazyimages"):
            seg_ranges = seg["ranges"]
            new_range = np.stack(
                (
                    np.searchsorted(tok_range[:, 0], seg_ranges[:, 0], 'left'),
                    np.searchsorted(tok_range[:, 1], seg_ranges[:, 1], 'right'),
                ),
                axis=-1,
            )
            seg['ranges'] = new_range

    return seg_data


def data_instance_2_feature(dconv, data_instance):
    doc2d = data_instance.document_2d
    hard_limit = dconv._max_seq_length \
                    if dconv._long_page_strategy == LongPageStrategy.FIRST_PART \
                    else None
    if dconv.skip_text_tokens:
        doc2d.tokens = []
    example = dconv.create_example(deepcopy(doc2d), hard_limit)
    spans = dconv.create_spans(example)
    doc_span = spans[0]
    if not doc_span.tokens:
        doc_span.seg_data['pages'] = example.seg_data['pages']

    startx, starty, endx, endy = doc_span.seg_data['pages']['bboxes'][0, :]
    if startx + starty != 0 or endx <= 0 or endy <= 0:
        logger.warning(f"Wrong page_bbox: [{startx} {starty} {endx} {endy}]! Skipping doc: {doc2d.docid}")
        return None
    span_decoder = dconv.convert_span_for_decoder(doc_span, data_instance.output, data_instance.input_prefix)
    feature = dconv.convert_span_to_feature(span_decoder)
    feature.doc_id = data_instance.identifier
    feature.label_name = data_instance.output_prefix
    return feature


# class TrainingT5DataConverter(DataConverter):
#     """Data converter for training/finetuning."""
#
#     def __init__(
#         self,
#         tokenizer: PreTrainedTokenizer,
#         masked_lm_prob: float = 0.2,
#         mean_noise_span_length: float = 3.5,
#         max_seq_length: int = 512,
#         long_page_strategy: LongPageStrategy = LongPageStrategy.RANDOM_PART,
#         img_matrix_order: int = 0,
#         **kwargs: Any,
#     ):
#         super().__init__(
#             tokenizer=tokenizer,
#             long_page_strategy=long_page_strategy,
#             max_seq_length=max_seq_length,
#             padding_idx=tokenizer.pad_token_id,
#             **kwargs,
#         )
#         self._mean_noise_span_length = mean_noise_span_length
#         self._long_page_strategy = long_page_strategy
#         self._masked_lm_prob = masked_lm_prob
#         self._img_matrix_order = img_matrix_order
#
#     def _create_t5_input_and_target(
#         self, bpe_tokens: List[str], seg_data
#     ) -> Tuple[Sequence[str], Sequence[int], Sequence[str], Dict[str, Any]]:
#
#         span_length = len(bpe_tokens)
#
#         if span_length > 5:
#             bpe_tokens = np.array(bpe_tokens)
#             if self._salient_spans:
#                 noise_mask, noise_spans_ranges = self._salient_noise_mask(bpe_tokens, seg_data)
#             else:
#                 noise_mask, noise_spans_ranges = random_spans_noise_mask(
#                     span_length, seg_data, self._masked_lm_prob, self._mean_noise_span_length
#                 )
#             input_tokens = noise_span_to_unique_sentinel(bpe_tokens, noise_mask, self.tokenizer)
#             target_tokens = noise_span_to_unique_sentinel(bpe_tokens, np.logical_not(noise_mask), self.tokenizer)
#             if target_tokens[-1].startswith("<extra"):
#                 target_tokens = target_tokens[:-1]
#             seg_data = _recompute_seg_data_for_sentinels(deepcopy(seg_data), noise_mask, noise_spans_ranges)
#         else:
#             target_tokens = input_tokens = bpe_tokens
#
#         # add special tokens for target for easier pretraining
#         target_tokens_padded = np.array(list(target_tokens) + [self.tokenizer.eos_token])
#
#         mask_indices = np.arange(len(target_tokens_padded))
#
#         return input_tokens, mask_indices, target_tokens_padded, seg_data
#
#     def _create_single_span(
#         self,
#         bpe_tokens: List[str],
#         org_tokens_idx: List[int],
#         token_label_ids: List[int],
#         seg_data: Dict[str, Any],
#         example_id: str,
#         span_idx: int,
#         start_position: int,
#         end_position: int,
#     ) -> Optional[Span]:
#
#         try:
#             masked_bpe_tokens, masked_lm_positions, masked_lm_labels, seg_data = self._create_t5_input_and_target(
#                 bpe_tokens, seg_data
#             )
#         except NotEnoughSalientSpans:
#             return None
#
#         order = self._img_matrix_order
#         if order > 0:
#             img_str = '<extra_id_99> ' * order ** 2
#             img_tokens_bpe = self.tokenizer.tokenize(img_str)
#             masked_bpe_tokens = np.concatenate((np.array(img_tokens_bpe), masked_bpe_tokens))
#             img_tokens_len = len(img_tokens_bpe)
#             org_tokens_idx = [-1] * img_tokens_len + org_tokens_idx
#
#             seg_data = deepcopy(seg_data)
#             startx, starty, endx, endy = seg_data['pages']['bboxes'][0, :]
#             assert startx + starty == 0 and endx > 0 and endy > 0
#             startx_img = np.mgrid[startx : endx : endx / order]
#             startx_img = np.tile(startx_img.reshape(order, 1), order).flatten()
#             endx_img = startx_img + endx / order
#             starty_img = np.mgrid[starty : endy : endy / order]
#             starty_img = np.tile(starty_img, order)
#             endy_img = starty_img + endy / order
#             img_bboxes = np.stack((startx_img, starty_img, endx_img, endy_img), axis=1)
#             seg_data['tokens']['bboxes'] = np.concatenate((img_bboxes, seg_data['tokens']['bboxes']), axis=0)
#             seg_data['tokens']['org_bboxes'] = np.concatenate((img_bboxes, seg_data['tokens']['org_bboxes']), axis=0)
#             seg_data['pages']['ranges'] = seg_data['pages']['ranges'] + img_tokens_len
#             excess_tokens_count = len(masked_bpe_tokens) - self._max_seq_length
#             if excess_tokens_count > 0:
#                 masked_bpe_tokens = masked_bpe_tokens[:-excess_tokens_count]
#                 org_tokens_idx = org_tokens_idx[:-excess_tokens_count]
#                 seg_data['tokens']['bboxes'] = seg_data['tokens']['bboxes'][:-excess_tokens_count]
#                 seg_data['tokens']['org_bboxes'] = seg_data['tokens']['org_bboxes'][:-excess_tokens_count]
#         span = Span(
#             example_id,
#             span_idx,
#             start_position,
#             end_position,
#             masked_bpe_tokens,
#             masked_lm_positions,
#             masked_lm_labels,
#             seg_data,
#             org_tokens_idx,
#             np.array([-1]),
#         )
#
#         return self.adjust_bboxes_to_page_number(span)


class T5DownstreamDataConverter(DataConverter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        long_page_strategy=LongPageStrategy.FIRST_PART,
        max_seq_length: int = 512,
        segment_levels: Tuple[str, ...] = ('tokens', 'lines'),
        overlap: int = 100,
        additional_bpe_tokens_count: int = 0,
        prefix_bbox_fill_value: float = -0.01,  # small negative value to easily distinguish prefixes
        img_matrix_order: int = 0,
        processes=1,
        imap_chunksize=1,
        skip_text_tokens=False,
        **kwargs: Any,
    ):
        super().__init__(
            tokenizer,
            long_page_strategy=long_page_strategy,
            max_seq_length=max_seq_length,
            segment_levels=segment_levels,
            overlap=overlap,
            additional_bpe_tokens_count=additional_bpe_tokens_count,
            padding_idx=tokenizer.pad_token_id,
            **kwargs,
        )
        if skip_text_tokens:
            logging.warning("You are using dataconverter with skip_text_tokens mode. "
                            "All text tokens will be removed from input")
        self.skip_text_tokens = skip_text_tokens
        self.prefix_bbox_fill_value = prefix_bbox_fill_value
        self.img_matrix_order = img_matrix_order
        self.processes = processes
        self.imap_chunksize = imap_chunksize

    def _add_special_tokens_to_segments(self, seg_data, start_ct, end_ct=0):
        """
        :param seg_data: dictionary of segment data to be modify in-place
        :param start_ct: number of tokens at the begining to add
        :param end_ct: number of tokens at the end to add
        :return: updated seg_data
        """
        # modify segment ranges and token bboxes
        fill_value = self.prefix_bbox_fill_value
        order = self.img_matrix_order
        img_tok_count = order ** 2
        startx, starty, endx, endy = seg_data['pages']['bboxes'][0, :]
        for segkey, seg in seg_data.items():
            if segkey == 'tokens':
                seg['bboxes'] = np.pad(
                    seg['bboxes'], ((start_ct, end_ct), (0, 0)), 'constant', constant_values=fill_value
                )
                seg['org_bboxes'] = np.pad(
                    seg['org_bboxes'], ((start_ct, end_ct), (0, 0)), 'constant', constant_values=fill_value
                )
                # check if special tokens should have some 2d positons
                x_special = (
                    np.arange(start_ct - img_tok_count) / 20 + fill_value
                )  # so 1 token is about 5% of page width
                seg['bboxes'][img_tok_count:start_ct, [0, 2]] = x_special[:, None]

                # add bboxes for image grid
                if order > 0:
                    startx_img = np.mgrid[startx : endx : endx / order]
                    startx_img = np.tile(startx_img.reshape(order, 1), order).flatten()
                    endx_img = startx_img + endx / order
                    starty_img = np.mgrid[starty : endy : endy / order]
                    starty_img = np.tile(starty_img, order)
                    endy_img = starty_img + endy / order
                    seg['bboxes'][:img_tok_count] = np.stack((startx_img, starty_img, endx_img, endy_img), axis=1)
            elif segkey == 'lines':
                seg['ranges'] = seg['ranges'] + start_ct
                # add line for prefix
                seg['ranges'] = np.concatenate((np.array([[0, start_ct]]), seg['ranges']))
                seg['bboxes'] = np.concatenate((np.array([[fill_value] * 4]), seg['bboxes']))
            elif segkey == 'pages':
                seg['ranges'] = seg['ranges'] + start_ct

        return seg_data

    def _add_special_tokens(
        self, bpe_tokens: List[str], org_tokens_idx: List[int], token_label_ids: List[int], seg_data: Dict[str, Any]
    ) -> Tuple[List[str], List[int], List[int], Dict[str, Any]]:
        return bpe_tokens, org_tokens_idx, token_label_ids, seg_data

    def generate_features(self, training_instances_iterator: Iterator[DataInstance]) -> Iterator[Feature]:
        func = partial(data_instance_2_feature, self)
        if self.processes > 1:
            with Pool(processes=self.processes) as pool:
                for feature in pool.imap(func, training_instances_iterator, chunksize=self.imap_chunksize):
                    if feature is not None:
                        yield feature
        # skip Pool for easier debugging
        else:
            for feature in map(func, training_instances_iterator):
                if feature is not None:
                    yield feature

    def convert_span_for_decoder(self, span, label_value, prefix, max_answer_length=1024):
        """
        :param span: span which need to be modified
        :param label_name: name of the labal which will be added to the input tokens
        :param label_value: value of the label which need to be predicted
        :param max_answer_length: limit of decoder answer length
        :return: modified span
        """

        prefix = '<extra_id_99> ' * self.img_matrix_order ** 2 + prefix
        tokenizer_ = self.tokenizer
        prefix_encoder_bpe = tokenizer_.tokenize(prefix)
        prefix_len = len(prefix_encoder_bpe)

        answer_bpe = tokenizer_.tokenize(label_value) + [tokenizer_.eos_token]
        if len(answer_bpe) > max_answer_length:
            logger.warning(f"Used max_answer_length={max_answer_length} cannot encode whole answer")
        answer_bpe = answer_bpe[:max_answer_length]

        tokens = prefix_encoder_bpe + span.tokens
        original_tokens_indices = [-1] * prefix_len + span.original_tokens_indices
        token_label_indices = [0] * prefix_len + span.token_label_indices
        copy_span = deepcopy(span)
        seg_data = self._add_special_tokens_to_segments(copy_span.seg_data, prefix_len)
        excess_tokens_count = len(tokens) - self._max_seq_length
        if excess_tokens_count > 0:
            tokens = tokens[:-excess_tokens_count]
            token_label_indices = token_label_indices[:-excess_tokens_count]
            original_tokens_indices = original_tokens_indices[:-excess_tokens_count]
            seg_data['tokens']['bboxes'] = seg_data['tokens']['bboxes'][:-excess_tokens_count]
            seg_data['tokens']['org_bboxes'] = seg_data['tokens']['org_bboxes'][:-excess_tokens_count]

        new_span = Span(
            example_id=span.example_id,
            span_index=span.span_index,
            start_position=span.start_position,
            end_position=span.end_position,
            tokens=tokens,
            masked_positions=list(np.arange(len(answer_bpe))),
            masked_labels=answer_bpe,
            seg_data=seg_data,
            original_tokens_indices=original_tokens_indices,
            token_label_indices=token_label_indices,
        )
        new_span = self.adjust_bboxes_to_page_number(new_span)
        return new_span
