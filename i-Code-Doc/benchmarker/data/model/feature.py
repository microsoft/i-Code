from typing import Any, Dict, Optional

import numpy as np


class Feature(object):
    """
        Feature representation.

        :param input_ids: input ids
        :param input_masks: input masks
        :param lm_label_ids: label ids
        :param seg_data: segmentation data
        :param token_label_ids: token label ids
        :param gold_words: store SINGLE gold word; name suggests multiple words but it was kept for backward compatibility
        :param masked_word_ids: store SINGLE masked word id; name suggest multiple masked words but it was kept for backward compatibility
        """

    def __init__(self, input_ids: np.ndarray, input_masks: np.ndarray, lm_label_ids: np.ndarray,
                 seg_data: Dict[str, Any], token_label_ids: np.ndarray,
                 gold_words: Optional[str] = None, masked_word_ids: Optional[int] = None):
        self.__input_ids = input_ids
        self.__input_masks = input_masks
        self.__lm_label_ids = lm_label_ids
        self.__seg_data = seg_data
        self.__token_label_ids = token_label_ids
        self.__gold_words = gold_words
        self.__masked_word_ids = masked_word_ids

    @property
    def input_ids(self) -> np.ndarray:
        return self.__input_ids

    @property
    def input_masks(self) -> np.ndarray:
        return self.__input_masks

    @property
    def lm_label_ids(self) -> np.ndarray:
        return self.__lm_label_ids

    @property
    def seg_data(self) -> Dict[str, Any]:
        return self.__seg_data

    @property
    def token_label_ids(self) -> np.ndarray:
        return self.__token_label_ids

    @property
    def gold_words(self) -> Optional[str]:
        return self.__gold_words

    @property
    def masked_word_ids(self) -> Optional[int]:
        return self.__masked_word_ids

    @gold_words.setter  # type: ignore
    def gold_words(self, value: Optional[str]):
        self.__gold_words = value

    @masked_word_ids.setter  # type: ignore
    def masked_word_ids(self, value: Optional[int]):
        self.__masked_word_ids = value

    def __repr__(self) -> str:
        return f'Feature[input_ids={self.input_ids}, input_masks={self.input_masks},' \
               f' lm_label_ids={self.lm_label_ids}, seg_data={self.seg_data}, ' \
               f' token_label_ids={self.token_label_ids}'

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise ValueError(f'Item not found: {item} :(')
