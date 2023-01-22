import math
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import torch
from transformers import PreTrainedTokenizer

from benchmarker.data.model.span import Span


def get_span_around_middle_point(middle_point: int,
                                 max_bpe_seq_length: int,
                                 upper_limit: int,
                                 lower_limit: int = 0
                                 ) -> Tuple[int, int]:
    from_range = max(
        lower_limit,
        min(
            middle_point - max_bpe_seq_length // 2,
            upper_limit - max_bpe_seq_length,
        ),
    )
    to_range = min(from_range + max_bpe_seq_length, upper_limit)
    return from_range, to_range


def get_span_within_page(page_ranges: List[List[int]],
                         middle_point: int,
                         max_bpe_seq_length: int
                         ) -> Tuple[int, int]:
    for page_range in page_ranges:
        page_lower_range, page_upper_range = map(int, page_range)
        if middle_point < page_lower_range or middle_point >= page_upper_range:
            continue
        from_range, to_range = get_span_around_middle_point(middle_point,
                                                            max_bpe_seq_length,
                                                            page_upper_range,
                                                            page_lower_range)
        return from_range, to_range
    raise ValueError(f'Random token: {middle_point} not in any page ranges.')


class BaseSlicer(metaclass=ABCMeta):
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 overlap: int = 0,
                 max_bpe_seq_length: int = 510):
        """
        Base constructor for slicers.

        While `overlap` and `tokenizer` parameters are not obligatory in all subclasses,
        having a unified constructor makes it easier to use different slicing strategies
        in other places.

        :param tokenizer: Tokenizer.
        :param overlap: Amount of overlap between subsequent slices.
        :param max_bpe_seq_length: Maximum number of BPE tokens in single span. Consider
        that usually single special token will be added to beginning and end of each
        span (hence the default value of 510).
        """
        self._tokenizer = tokenizer
        self._overlap = overlap
        self._max_bpe_seq_length = max_bpe_seq_length

    @abstractmethod
    def create_slices(
        self, tokens: Sequence[str], seg_data: Optional[Mapping[str, Any]]
    ) -> Sequence[Tuple[int, int]]:
        """
        Slices example into span ranges based on its token list and segment data.

        It returns only ranges (instead of sliced data), as creating spans should be
        DataConverter's job and it could vary depending on the implementation.

        :param tokens: List of tokens
        :param seg_data: Segment data.
        :return: List of span range indices.
        """

    @abstractmethod
    def join_predictions(
        self,
        model_output: Union[torch.Tensor, Sequence[torch.Tensor]],
        span_data: Sequence[Span],
    ) -> torch.Tensor:
        """
        Joins span's predictions into single prediction.

        :param model_output: List of tensors for spans of dim: seq_length x features_dim, or tensor
        shaped num_spans x seq_length x features_dim
        :param span_data: Original span data for given example. Preferably output of
        DataConverter.create_spans
        :return: Tensor of shape example_length x features_dim
        """


class AverageJoinSlicer(BaseSlicer):
    """
    Base class for slicers.

    Implements general joining method which should be suitable for most slicing
    algorithms, since it uses data from resulting slices to determine how they
    should be joined.

    Each pair of following slices have their overlap calculated independently,
    and averaging weights are calculated using cosine function with values scaled
    to [0,1].
    """

    @staticmethod
    def _get_total_sequence_length(span_data: Sequence[Span]) -> int:
        return span_data[-1].end_position

    @staticmethod
    def _get_averaging_weights(length: int) -> torch.Tensor:
        return (1 + torch.cos(torch.linspace(0, math.pi, steps=length))) / 2

    def _generate_slice_joining_weights(
        self,
        size: Tuple,
        left_overlaps: List[int],
        right_overlaps: List[int],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        weights = torch.ones(size, device=device, dtype=dtype)
        for i, (left_overlap, right_overlap) in enumerate(
            zip(left_overlaps, right_overlaps)
        ):
            if left_overlap > 0:
                weights[i, :left_overlap] = - self._get_averaging_weights(left_overlap) + 1
            if right_overlap > 0:
                weights[i, -right_overlap:] = self._get_averaging_weights(right_overlap)
        return weights

    def join_predictions(
        self,
        model_output: Union[torch.Tensor, Sequence[torch.Tensor]],
        span_data: Sequence[Span]
    ) -> torch.Tensor:
        """
        Joins predictions with weighted averaging of overlapping parts.

        Assumes that output for any special tokens is already removed from model_output.

        :param model_output: List of tensors for spans of dim: seq_length x features_dim, or tensor
        shaped num_spans x seq_length x features_dim
        :param span_data: Original span data for given example. Preferably output of
        DataConverter.create_spans.
        :return: Tensor of shape example_length x features_dim
        """
        device = model_output[0].device
        dtype = model_output[0].dtype

        total_length = self._get_total_sequence_length(span_data)
        joined_output = torch.zeros(total_length, model_output[0].size(-1),
                                    device=device, dtype=dtype)
        overlaps = [0]
        for i, (preceding_span, succeding_span) in enumerate(
            zip(span_data[:-1], span_data[1:])
        ):
            overlap = preceding_span.end_position - succeding_span.start_position
            overlaps.append(overlap)
        overlaps.append(0)
        joining_weights = self._generate_slice_joining_weights(
            (len(model_output), max([sp.size(0) for sp in model_output])),  # type: ignore
            overlaps[:-1],
            overlaps[1:],
            device=device,
            dtype=dtype
        ).unsqueeze(2)
        for span_wght, span_emb, span in zip(joining_weights, model_output,  # type: ignore
                                             span_data):
            wght_span_emb = span_wght[:len(span_emb)] * span_emb
            span_length = span.end_position - span.start_position
            joined_output[
                span.start_position:span.end_position
            ] += wght_span_emb[:span_length]

        return joined_output

    def create_slices(
        self, tokens: Sequence[str], seg_data: Optional[Mapping[str, Any]] = None
    ) -> Sequence[Tuple[int, int]]:
        non_zero_slices = []
        for span in self._create_slices(tokens, seg_data):
            if span[1] > span[0]:
                non_zero_slices.append(span)
        return non_zero_slices

    def _create_slices(
        self, tokens: Sequence[str], seg_data: Optional[Mapping[str, Any]]
    ) -> Sequence[Tuple[int, int]]:
        raise NotImplementedError


class FirstPartSlicer(AverageJoinSlicer):
    """
    Extracts only first part from each example.
    """

    def _create_slices(
        self, tokens: Sequence[str], seg_data: Optional[Mapping[str, Any]] = None
    ) -> Sequence[Tuple[int, int]]:
        from_range = 0
        to_range = min(self._max_bpe_seq_length, len(tokens))
        return [(from_range, to_range)]


class AllPartsInPagesSlicer(AverageJoinSlicer):
    def _create_slices(
            self,  tokens: Sequence[str], seg_data: Optional[Mapping[str, Any]]
    ) -> Sequence[Tuple[int, int]]:
        assert seg_data is not None
        page_ranges = seg_data['pages']['ranges']
        num_tokens = len(tokens)
        assert page_ranges[-1][1] == num_tokens
        assert (
            self._overlap < self._max_bpe_seq_length
            or num_tokens <= self._max_bpe_seq_length
        )
        splits: List[Tuple[int, int]] = []

        for page_range in page_ranges:
            page_lower_range, page_upper_range = map(int, page_range)
            splits += [
                (a, min(a + self._max_bpe_seq_length, page_upper_range))
                for a in range(
                    page_lower_range,
                    max(page_upper_range - self._overlap, page_lower_range + 1),
                    self._max_bpe_seq_length - self._overlap,
                )
            ]
        return splits


class LongPageStrategy(str, Enum):
    FIRST_PART = 'FIRST_PART'
    ALL_PARTS_IN_PAGES = 'ALL_PARTS_IN_PAGES'           # FIXME: zostawic

    @property
    def constructor_mapping(self) -> Dict['LongPageStrategy', Type]:
        return {  # type: ignore
            self.FIRST_PART: FirstPartSlicer,  # type: ignore
            self.ALL_PARTS_IN_PAGES: AllPartsInPagesSlicer,  # type: ignore
        }

    def create_slicer(
        self,
        overlap: Optional[int] = None,
        max_bpe_seq_length: Optional[int] = None,
        **kwargs: Optional[Any]
    ) -> BaseSlicer:
        if overlap is not None:
            kwargs.update(overlap=overlap)
        if max_bpe_seq_length is not None:
            kwargs.update(max_bpe_seq_length=max_bpe_seq_length)
        return cast(BaseSlicer, self.constructor_mapping[self](**kwargs))
