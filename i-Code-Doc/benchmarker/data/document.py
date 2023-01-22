from typing import Any, Dict, Optional, Sequence

import numpy as np

from benchmarker.utils.cmp_helpers import nested_dict_with_arrays_cmp


class Doc2d:
    """
    :param docid: Document id
    :param tokens: List of tokens in document
    :param token_ocr_ranges: if this is required,
        document can store also character spans of tokens in document
    :param seg_data: Dictionary of visual objects used by 2D models
    :param token_label_ids: store label_ids of each token, used for token classification tasks
    """

    def __init__(
        self,
        tokens: Sequence[str],
        seg_data: Dict[str, Any],
        token_ocr_ranges: Optional[np.ndarray] = None,
        token_label_ids: Sequence[int] = None,
        docid: str = '',
    ):
        self.docid = docid
        self.token_ocr_ranges = token_ocr_ranges
        self.tokens = tokens
        self.seg_data = seg_data
        self.token_label_ids = token_label_ids

    def __len__(self) -> int:
        return len(self.tokens)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Doc2d)
            and self.docid == other.docid
            and self.tokens == other.tokens
            and np.all(self.token_ocr_ranges == other.token_ocr_ranges)
            and self.token_label_ids == other.token_label_ids
            and nested_dict_with_arrays_cmp(self.seg_data, other.seg_data)
        )
