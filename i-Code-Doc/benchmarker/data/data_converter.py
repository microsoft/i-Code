import collections
import logging
from abc import ABCMeta
from copy import deepcopy
from itertools import accumulate
from random import random, sample
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from benchmarker.data.document import Doc2d
from benchmarker.data.model.example import Example
from benchmarker.data.model.feature import Feature
from benchmarker.data.model.span import Span
from benchmarker.data.slicer import BaseSlicer, LongPageStrategy
from benchmarker.data.utils import (
    FEAT_META,
    convert_to_np,
    fix_missing_tokens_in_lines,
    get_bpe_positions,
    get_data_part, single_line_spans
)


class EmptyBPETokensException(BaseException):
    """Raise when tokenizing document returns empty list."""


class NotEnoughSalientSpans(BaseException):
    """Raise when there are not enough salient spans to mask"""


class DataConverter(metaclass=ABCMeta):
    """Base class for converting documents into proper datamodel. We can use different conversion
    strategies for pages with many tokens (above max_seq_len parameter in original BERT model)
    to specific usecases: preparation data for training, word-gap challenge, vectorization, etc.:
     * 'RANDOM_PART' - choose random part of a document (use for model training)
     * 'FIRST_PART' - choose first part of a document
        (use for model training or document/page classification)
     * 'ALL_PARTS' - take into account all parts of document with with some overlaps
        (can be used for most of usecases)
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer = None,
                 long_page_strategy: LongPageStrategy = LongPageStrategy.ALL_PARTS_IN_PAGES,
                 max_seq_length: int = 512,
                 segment_levels: Tuple[str, ...] = ('tokens', 'lines'),
                 overlap: int = 0,
                 additional_bpe_tokens_count: int = 2,
                 padding_idx: Optional[int] = None,
                 salient_spans: bool = False,
                 **kwargs: Any):
        self._padding_idx = padding_idx
        self._segment_levels = segment_levels
        self._toklevel = 'tokens' in segment_levels
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._max_bpe = max_seq_length - additional_bpe_tokens_count
        self._long_page_strategy = long_page_strategy
        self._segment_levels_cleaned = set([s for s in segment_levels if s != 'tokens'])
        self._overlap = overlap
        self._salient_spans = salient_spans
        if self._salient_spans:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except ImportError:
                "Please install spacy lib in order to use salient spans masking"

        self._slicer = self._long_page_strategy.create_slicer(overlap=overlap,
                                                              max_bpe_seq_length=self._max_bpe,
                                                              tokenizer=self.tokenizer)
        self._example_counter = 0

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Data tokenizer"""
        return self._tokenizer

    @property
    def slicer(self) -> BaseSlicer:
        """Example slicer"""
        return self._slicer

    @property
    def segment_levels(self) -> Tuple[str, ...]:
        """Segment levels, it could be one of:
            * 'tokens'
            * 'lines'
            * 'pages'
        """
        return self._segment_levels

    @property
    def max_seq_length(self) -> int:
        """Max sequence length for the model."""
        return self._max_seq_length

    def _create_input_and_target(self, bpe_tokens: List[str], seg_data: Dict[str, Any] = None) \
            -> Tuple[Sequence[str], Sequence[int], Sequence[str]]:
        """For TokenClassificationDataConverter there is no need to mask tokens
        :param seg_data:
        """
        return bpe_tokens, [], []

    def convert_doc2d_to_spans(self, doc: Doc2d) -> Sequence[Span]:
        """
            Methods for conversion page to model compatible spans

            :param doc: Doc2d instance

            :return: list of spans (one span = dict of page part elements)
        """
        return self.create_spans(self.create_example(doc))

    def _salient_noise_mask(self, bpe_tokens: np.ndarray, seg_data: Dict[str, Any],
                            min_salient_accepted_scalar=0.4, max_salient_accepted_scalar=1.2,
                            balancing_coef=0.7, min_no_of_spans=5):
        """

        :param bpe_tokens: numpy array of bpe tokens
        :param seg_data: seg_data used for optional line boundary of salient spans
        :param min_salient_accepted_scalar: together with _masked_lm_prob define minimum of masked tokens
        :param max_salient_accepted_scalar: together with _masked_lm_prob define maximum of masked tokens
        :param balancing_coef: used for limiting number of spans for frequent entity categories
        :param min_no_of_spans: should be at least of this number of spans selected to continue of document processing

        :return noise mask and ranges of salient spans

        """
        length = len(bpe_tokens)
        # check how space is tokenized
        assert self.tokenizer.tokenize(" a")[0][0] in ('▁', 'Ġ'), "salient feature does not support this tokenizer"
        text = "".join(bpe_tokens).replace('▁', ' ').replace('Ġ', ' ')

        starts = np.cumsum([0] + [len(tok) for tok in bpe_tokens])
        # check if last token ranges are correct
        assert text[starts[-2]:starts[-1]].strip() == bpe_tokens[-1].replace('▁', ' ').replace('Ġ', ' ').strip(), \
            "Text for salient span processing was built incorrectly"
        doc = self.nlp(text)

        doc_entities = collections.defaultdict(list)
        for ent in doc.ents:
            if ent.start_char == 0:
                continue
            doc_entities[ent.label_].append(ent)

        # compute probabilities of using span
        category_sample = {k: round(len(v)**balancing_coef) for k, v in doc_entities.items()}
        selected_entities = {k: sample(v, category_sample[k]) for k, v in doc_entities.items()}
        flat_entities = [ent for ents in selected_entities.values() for ent in ents]

        if len(flat_entities) < min_no_of_spans:
            raise NotEnoughSalientSpans("Skipping span")

        char_ranges = np.sort([(ent.start_char, ent.end_char) for ent in flat_entities], 0)

        span_starts = np.searchsorted(starts, char_ranges[:, 0], "right") - 1
        span_ends = np.searchsorted(starts, char_ranges[:, 1], "left")
        span_lengths = span_ends - span_starts
        noise_len_goal = length * self._masked_lm_prob

        span_ranges = np.stack([span_starts, span_ends], -1)

        # clean-up ranges
        # first token should not be masked + should not be two consecutive spans being masked
        idx_clean = []
        last_end = -1
        for rng_idx, rng in enumerate(span_ranges):
            if rng[0] == 0 or rng[0] <= last_end:
                idx_clean.append(False)
            else:
                idx_clean.append(True)
                last_end = rng[1]
        span_ranges = span_ranges[idx_clean]
        span_lengths = span_lengths[idx_clean]

        if span_lengths.sum() > noise_len_goal * max_salient_accepted_scalar:
            # drop some of the spans in random order
            drop_goal = span_lengths.sum() - noise_len_goal
            dropped_idx = []
            dropped_len = 0
            for idx, single_span_len in sorted(enumerate(span_lengths), key=lambda x: random()):
                dropped_idx.append(idx)
                dropped_len += single_span_len
                if dropped_len >= drop_goal:
                    break
            span_ranges = np.delete(span_ranges, dropped_idx, axis=0)
            span_lengths = np.delete(span_lengths, dropped_idx, axis=0)

        if span_lengths.sum() < noise_len_goal * min_salient_accepted_scalar:
            # skip span processing
            raise NotEnoughSalientSpans("Skipping span")

        spans_sorted = np.sort(span_ranges, 0)
        spans_sorted = single_line_spans(spans_sorted, seg_data)
        noise_mask = np.zeros(length, dtype=np.bool)
        for rng in spans_sorted:
            noise_mask[rng[0]:rng[1]] = True

        return noise_mask, spans_sorted

    def create_example(self, doc: Doc2d, hard_token_limit: Optional[int] = None) -> Example:
        """Create example based as an input to the model."""
        # fix documents with missing tokens in lines
        doc = fix_missing_tokens_in_lines(doc)
        tokens, token_ocr_ranges, seg_data = \
            doc.tokens, doc.token_ocr_ranges, doc.seg_data

        if hard_token_limit is not None:
            tokens = tokens[:hard_token_limit]
            if token_ocr_ranges is not None:
                token_ocr_ranges = token_ocr_ranges[:hard_token_limit]

        if doc.token_label_ids is None:
            token_label_ids: Sequence[int] = [0] * len(tokens)
        else:
            token_label_ids = doc.token_label_ids

        # transform to bpe
        bpe_tokens, bpe_token_ocr_ranges, seg_data, org_tokens_idx, tok_bpe_map = \
            self._transform_to_bpe(tokens, token_ocr_ranges, seg_data)
        # add normalized bboxes
        page_same_size, page_most_dim = self._get_page_dim(seg_data['pages']['org_bboxes'])
        for skey, seg in seg_data.items():
            if skey in ("images", "lazyimages"):
                continue
            seg['bboxes'] = self._convert_to_relative_page_pos(seg, skey, seg_data['pages'],
                                                               page_same_size, page_most_dim)
        # add page numbers
        seg_data = self._add_page_no(seg_data)

        bpe_token_label_ids = [token_label_ids[i] for i in org_tokens_idx]

        example_id = str(self._get_exid()) if doc.docid is None else doc.docid  # type: ignore

        return Example(example_id, bpe_tokens, bpe_token_ocr_ranges, org_tokens_idx, tok_bpe_map,
                       seg_data, bpe_token_label_ids)

    def create_spans(self, ex: Example) -> Sequence[Span]:
        """Create spans for the example."""
        max_bpe = self._max_bpe
        spans = []

        # apply long page strategies
        for span_idx, (from_range, to_range) in enumerate(self._get_splits(ex.tokens, ex.seg_data)):
            part_tokens, part_org_tokens_idx, part_token_label_ids, part_seg_data = \
                get_data_part(from_range,
                              to_range,
                              max_bpe,
                              ex.tokens,
                              list(ex.original_token_indices),
                              list(ex.token_label_indices),
                              ex.seg_data)
            spans.append(
                    self._create_single_span(part_tokens,
                                             part_org_tokens_idx,
                                             part_token_label_ids,
                                             part_seg_data,
                                             ex.example_id,
                                             span_idx,
                                             from_range,
                                             to_range)
            )

        return [span for span in spans if span is not None and self._is_span_valid(span)]

    def convert_span_to_feature(self, span: Span, seq_len: Optional[int] = None) -> Feature:
        """Convert span to feature - input for the model."""

        seq_len = self._max_seq_length if seq_len is None else seq_len
        assert len(span.tokens) <= seq_len, 'The preprocessed data should be already truncated'

        # compute 1D features
        input_ids_raw = self.tokenizer.convert_tokens_to_ids(span.tokens)
        input_ids = self._feature_empty_array('input_ids', seq_len)
        input_ids[:len(input_ids_raw)] = input_ids_raw

        input_masks = self._feature_empty_array('input_masks', seq_len)
        input_masks[:len(input_ids_raw)] = 1

        masked_label_ids = self.tokenizer.convert_tokens_to_ids(span.masked_labels)
        label_ids = self._feature_empty_array('lm_label_ids', seq_len,
                                              default=self._padding_idx)
        label_ids[span.masked_positions] = masked_label_ids

        token_label_ids = self._feature_empty_array('token_label_ids', seq_len)
        token_label_ids[input_masks] = span.token_label_indices

        # add 2D features as a dictionary
        feat_seg = self.__add_2d_features_as_dict(seq_len, span)

        return Feature(input_ids, input_masks, label_ids, feat_seg, token_label_ids)

    def __add_2d_features_as_dict(self, seq_length: int, span: Span) -> Dict[str, Any]:
        feat_seg: Dict[str, Any] = {}
        for seg_key in self._segment_levels:
            seg = span.seg_data[seg_key]
            feat_seg[seg_key] = self.__add_segment_2d_features(seg, seq_length)
        return feat_seg

    def __add_segment_2d_features(self, seg: Dict[str, Any], seq_len: int) -> Dict[str, Any]:
        result = {}
        for el_key, el in seg.items():
            if el_key in FEAT_META.keys() and ("train_dtype" in FEAT_META[el_key]
                                               or el_key == "path"):
                if FEAT_META[el_key]["wide"]:
                    # create empty array using feature metadata
                    arr = self._feature_empty_array(el_key, seq_len)
                    arr[:len(el)] = el
                elif el_key == "img_data":
                    arr = self._feature_empty_array(el_key, seq_len)
                    try:
                        arr[:el.shape[0], :el.shape[1]] = el
                    except:
                        pass
                else:
                    arr = el
                result[el_key] = arr
            # for segments other than tokens create masks and segment_token_ids
            if el_key == 'ranges':
                mask_array = self._feature_empty_array('masks', seq_len)
                mask_array[:len(el)] = 1
                result['masks'] = mask_array
                token_map = self._feature_empty_array('token_map', seq_len)
                for idx, rng in enumerate(el):
                    token_map[rng[0]:rng[1]] = idx
                result['token_map'] = token_map
        return result

    @staticmethod
    def adjust_bboxes_to_page_number(span: Span) -> Span:
        cum_height = 0
        for height, (start, end) in zip(span.seg_data['pages']['bboxes'][:-1, 3],
                                        span.seg_data['pages']['ranges'][1:]):
            cum_height += height
            span.seg_data['tokens']['bboxes'][start:end, [1, 3]] += cum_height
            if "lines" in span.seg_data:
                rng = span.seg_data['lines']['ranges']
                rng_start = np.searchsorted(rng[:, 0], start, 'left')
                rng_end = np.searchsorted(rng[:, 1] - 1, end, 'right')
                span.seg_data['lines']['bboxes'][rng_start:rng_end, [1, 3]] += cum_height

        return span

    def convert_spans_to_features(self, spans: Sequence[Span],
                                  seq_length: Optional[int] = None) -> Sequence[Feature]:
        """Convert spans to features - inputs for the model"""
        return [self.convert_span_to_feature(span, seq_length) for span in spans]

    def _get_splits(self, bpe_tokens: Sequence[str],
                    seg_data: Dict[str, Any]) -> Sequence[Tuple[int, int]]:
        splits = self.slicer.create_slices(bpe_tokens, seg_data)
        return splits if splits else [(0, 0)]

    @staticmethod
    def _calculate_bpe_length(bpe_token: str) -> int:
        if bpe_token.startswith('##'):  # bert
            return len(bpe_token) - 2
        if bpe_token.startswith('Ġ'):  # roberta
            return len(bpe_token) - 1
        if bpe_token.startswith('▁'):  # xlm
            return len(bpe_token) - 1
        return len(bpe_token)

    def _transform_to_bpe(self, tokens: Sequence[str],
                          token_ocr_ranges: Optional[Sequence[Tuple[int, int]]],
                          seg_data: Dict[str, Any]) -> Tuple[Sequence[str],
                                                             np.ndarray,
                                                             Dict[str, Any],
                                                             Sequence[int],
                                                             Sequence[Tuple[int, int]]]:
        use_ocr_ranges = token_ocr_ranges is not None

        if isinstance(self._tokenizer, PreTrainedTokenizerFast):
            bpe = [e.tokens for e
                   in self._tokenizer.backend_tokenizer.encode_batch(tokens, add_special_tokens=False)]
        elif hasattr(self._tokenizer, 'add_prefix_space'):
            bpe = [self._tokenizer.tokenize(tok, add_prefix_space=True) if len(tok) > 0 else []
                   for tok in tokens]
        else:
            bpe = [self._tokenizer.tokenize(tok) if len(tok) > 0 else [] for tok in tokens]

        bpe_tokens: List[str] = []
        bpe_bboxes: List[Sequence[float]] = []
        org_tokens_idx = []
        tok_bpe_map = []
        if use_ocr_ranges:
            bpe_token_ocr_ranges = []  # type: ignore
        else:
            bpe_token_ocr_ranges = None  # type: ignore

        for tok_idx, (t, blst) in enumerate(zip(tokens, bpe)):
            curr_len = len(bpe_tokens)
            tok_bpe_map.append((curr_len, curr_len + len(blst)))
            bpe_tokens.extend(blst)
            org_tokens_idx.extend([tok_idx for _ in range(len(blst))])
            bpe_lens = [self._calculate_bpe_length(t) for t in blst]
            if 'tokens' in seg_data:
                bpe_bboxes.extend(
                    get_bpe_positions(seg_data['tokens']['org_bboxes'][tok_idx], bpe_lens))
            if use_ocr_ranges:
                cuml = [0] + list(accumulate(bpe_lens))
                trng = token_ocr_ranges[tok_idx]  # type: ignore
                bpe_token_ocr_ranges.extend((min(trng[0] + cuml[li], trng[1]),  # type: ignore
                                             min(trng[0] + cuml[li + 1], trng[1]))
                                            for li in range(len(bpe_lens)))

        if bpe_token_ocr_ranges is not None:
            bpe_token_ocr_ranges = convert_to_np(bpe_token_ocr_ranges, 'ocr_ranges').clip(0)

        # recalculate ranges in segments due to bpe, assign new bpe token boxes to dict item
        for skey, seg in seg_data.items():
            if skey == 'tokens':
                seg_data['tokens']['org_bboxes'] = convert_to_np(bpe_bboxes, 'org_bboxes')
            elif skey in ("lines", "pages"):
                seg['ranges'] = self._recalculate_seg_ranges(seg['ranges'], tok_bpe_map,
                                                             org_tokens_idx)

        return bpe_tokens, bpe_token_ocr_ranges, seg_data, org_tokens_idx, tok_bpe_map

    def _add_special_tokens(self, bpe_tokens: List[str], org_tokens_idx: List[int],
                            token_label_ids: List[int], seg_data: Dict[str, Any]) \
            -> Tuple[List[str], List[int], List[int], Dict[str, Any]]:
        # add special tokens
        bpe_tokens = [self.tokenizer.cls_token] + bpe_tokens \
                                    + [self.tokenizer.sep_token]
        org_tokens_idx = [-1] + org_tokens_idx + [-1]
        token_label_ids = [-1] + token_label_ids + [-1]

        # modify segment ranges and token bboxes
        for segkey, seg in seg_data.items():
            if segkey == 'tokens':
                seg['bboxes'] = np.pad(seg['bboxes'], ((1, 1), (0, 0)), 'constant',
                                       constant_values=0)
                seg['org_bboxes'] = np.pad(seg['org_bboxes'], ((1, 1), (0, 0)), 'constant',
                                           constant_values=0)
            elif segkey in ("lines", "pages"):
                seg['ranges'] = seg['ranges'] + 1

        return bpe_tokens, org_tokens_idx, token_label_ids, seg_data

    def _create_single_span(self, bpe_tokens: List[str], org_tokens_idx: List[int],
                            token_label_ids: List[int], seg_data: Dict[str, Any],
                            example_id: str, span_idx: int,
                            start_position: int, end_position: int) -> Span:

        bpe_tokens, org_tokens_idx, token_label_ids, seg_data = \
            self._add_special_tokens(bpe_tokens, org_tokens_idx, token_label_ids, seg_data)

        try:
            masked_bpe_tokens, masked_lm_positions, masked_lm_labels = \
                self._create_input_and_target(bpe_tokens, seg_data)
        except NotEnoughSalientSpans:
            return None

        return Span(example_id, span_idx, start_position, end_position, masked_bpe_tokens,
                    masked_lm_positions, masked_lm_labels, seg_data, org_tokens_idx,
                    token_label_ids)

    def _is_span_valid(self, span: Span) -> bool:
        tokens_len = len(span.tokens)
        valid = tokens_len <= self._max_seq_length and \
            len(span.token_label_indices) <= self._max_seq_length and \
            len(span.masked_positions) == len(span.masked_labels) and \
            len(span.masked_labels) <= self._max_seq_length and \
            all([self.__is_seg_valid(k, v, tokens_len) for k, v in span.seg_data.items()])
        if not valid:
            logging.warning(f"Span in document {span.example_id} seems to be corrupted")

        return valid

    def __is_seg_valid(self, key: str, seg: Dict[str, Any], token_count: int):
        if key == 'tokens':
            return token_count == len(seg['bboxes']) == len(seg['org_bboxes'])
        elif key in ("lines", "pages"):
            return len(seg['ranges']) == len(seg['bboxes']) == len(seg['org_bboxes']) and \
                   len(seg['ranges']) <= self._max_seq_length
        elif key in ("images", "lazyimages"):
            return True

    def _get_exid(self) -> int:
        curr_id = self._example_counter
        self._example_counter += 1
        return curr_id

    @staticmethod
    def _convert_to_relative_page_pos(seg: Dict[str, Any], seg_key: str,
                                      page_seg: Dict[str, Any], page_same_size: bool,
                                      page_most_dim: Tuple[int, ...]) -> np.ndarray:
        # positions - n x 4 array
        # page_positions = 1 x 4 array
        rel_pos = np.zeros_like(seg['org_bboxes'], dtype=float)
        rel_pos[:, 0] = seg['org_bboxes'][:, 0].clip(min=0, max=page_most_dim[2]) / (page_most_dim[2] * np.sqrt(2))
        rel_pos[:, 2] = seg['org_bboxes'][:, 2].clip(min=0, max=page_most_dim[2]) / (page_most_dim[2] * np.sqrt(2))
        rel_pos[:, 1] = seg['org_bboxes'][:, 1].clip(min=0, max=page_most_dim[3]) / (page_most_dim[2] * np.sqrt(2))
        rel_pos[:, 3] = seg['org_bboxes'][:, 3].clip(min=0, max=page_most_dim[3]) / (page_most_dim[2] * np.sqrt(2))

        if not page_same_size:
            for psize, prange in zip(page_seg['org_bboxes'], page_seg['ranges']):
                if tuple(psize) == page_most_dim:
                    continue
                # get index of bboxes which need to be scaled with different dimension
                if seg_key == 'tokens':
                    idx = list(range(prange[0], prange[1]))
                else:
                    idx = []
                    for seg_idx, seg_rng in enumerate(seg['ranges']):
                        if prange[0] <= seg_rng[0] < prange[1]:
                            idx.append(seg_idx)

                rel_pos[idx, 0] = seg['org_bboxes'][idx, 0].clip(min=0, max=psize[2]) / (psize[2] * np.sqrt(2))
                rel_pos[idx, 2] = seg['org_bboxes'][idx, 2].clip(min=0, max=psize[2]) / (psize[2] * np.sqrt(2))
                rel_pos[idx, 1] = seg['org_bboxes'][idx, 1].clip(min=0, max=psize[3]) / (psize[2] * np.sqrt(2))
                rel_pos[idx, 3] = seg['org_bboxes'][idx, 3].clip(min=0, max=psize[3]) / (psize[2] * np.sqrt(2))

        return rel_pos

    @staticmethod
    def _recalculate_seg_ranges(seg_ranges: np.ndarray, tok_bpe_map: Sequence[Tuple[int, int]],
                                org_tokens_idx: Sequence[int]) -> np.ndarray:
        if len(tok_bpe_map) == 0:
            bpe_rng_np = np.zeros((len(seg_ranges), 2), dtype=np.int)
        elif seg_ranges is not None:
            tok_idx = np.array(org_tokens_idx)
            bpe_rng_np = np.stack((np.searchsorted(tok_idx, seg_ranges[:, 0], 'left'),
                                   np.searchsorted(tok_idx, seg_ranges[:, 1] - 1, 'right')),
                                  axis=-1)
        return bpe_rng_np

    @staticmethod
    def _add_page_no(seg_data: Dict) -> Dict:
        """
        :param seg_data: dictionary of segments elements
        :return: dictionary of seg_data with added page elements
        """
        if "pages" in seg_data:
            seg_data_ = deepcopy(seg_data)
            seg_data_["pages"]["ordinals"] = np.arange(seg_data["pages"]["ranges"].shape[0])
            seg_data_["pages"]["cardinality"] = seg_data["pages"]["ranges"].shape[0]
            return seg_data_
        else:
            raise ValueError("pages is mandatory element of seg_data")

    @staticmethod
    def _feature_empty_array(feature: str, seq_length: int, default=None) -> np.ndarray:
        feat = FEAT_META[feature]
        if default is None:
            default = feat['default']
        return np.full(([seq_length] + feat['dim']) if feat["wide"] else feat['dim'],
                       dtype=feat['dtype'],
                       fill_value=default)

    @staticmethod
    def _get_page_dim(pages_dim: Sequence[Sequence[int]]) -> Tuple[bool, Tuple[int, ...]]:
        # function will return True if all pages are with the same dimension
        # and dimension of mostly use dimension in the document
        tpages = [tuple(a) for a in pages_dim]
        counter = collections.Counter(tpages)

        return len(counter) == 1, counter.most_common(1)[0][0]

