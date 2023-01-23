from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class Example(object):

    def __init__(self, example_id: str, tokens: Sequence[str], token_ocr_ranges: np.ndarray,
                 original_token_indices: Sequence[int], tokens_bpe_map: Sequence[Tuple[int, int]],
                 seg_data: Dict[str, Any], token_label_indices: Sequence[int]):
        """
        Construct Example data class.

        :param example_id: Example id
        :param tokens: Tokens
        :param token_ocr_ranges: OCR ranges
        :param original_token_indices: Original indices
        :param tokens_bpe_map: BPE map
        :param seg_data: Seg data
        :param token_label_indices: Label indices
        """
        self.__example_id = example_id
        self.__tokens = tokens if isinstance(tokens, list) else list(tokens)
        self.__token_ocr_ranges = token_ocr_ranges
        self.__original_token_indices = original_token_indices
        self.__tokens_bpe_map = tokens_bpe_map
        self.__seg_data = seg_data
        self.__token_label_indices = token_label_indices

    @property
    def example_id(self) -> str:
        """
        Return example id
        :return: Example id
        """
        return self.__example_id

    @property
    def tokens(self) -> List[str]:
        """
        Return tokens
        :return: Tokens
        """
        return self.__tokens

    @property
    def token_ocr_ranges(self) -> np.ndarray:
        """
        Return OCR ranges
        :return: OCR ranges
        """
        return self.__token_ocr_ranges

    @property
    def original_token_indices(self) -> Sequence[int]:
        """
        Return original token indices
        :return: Original token indices
        """
        return self.__original_token_indices

    @property
    def tokens_bpe_map(self) -> Sequence[Tuple[int, int]]:
        """
        Return BPE map
        :return: BPE map
        """
        return self.__tokens_bpe_map

    @property
    def seg_data(self) -> Dict[str, Any]:
        """
        Return seg data
        :return: Seg data
        """
        return self.__seg_data

    @property
    def token_label_indices(self) -> Sequence[int]:
        """
        Return label indices
        :return: Label indices
        """
        return self.__token_label_indices

    def __repr__(self):
        return f'Example[example_id={self.example_id}, tokens={self.tokens},' \
               f' token_ocr_ranges={self.token_ocr_ranges},' \
               f' original_token_indices={self.original_token_indices},' \
               f' tokens_bpe_map={self.tokens_bpe_map}, seg_data={self.seg_data},' \
               f' token_label_indices={self.token_label_indices}]'
