from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from benchmarker.data.document import Doc2d

IMG_SIZE = (384, 512)
IMG_SIZE_DIVISIBILITY = 64
FEAT_META = {
    'input_ids': {'dtype': np.int32, 'dim': [], 'wide': True, 'default': 0, 'train_dtype': torch.int64},
    'lm_label_ids': {'dtype': np.int32, 'dim': [], 'wide': True, 'default': -1, 'train_dtype': torch.int64},
    'labels': {'dtype': np.int32, 'dim': [], 'wide': True, 'default': -1, 'train_dtype': torch.int64},
    'input_masks': {'dtype': np.bool, 'dim': [], 'wide': True, 'default': 0, 'train_dtype': torch.uint8},
    'attention_mask': {'dtype': np.bool, 'dim': [], 'wide': True, 'default': 0, 'train_dtype': torch.uint8},
    'bboxes': {'dtype': np.float16, 'dim': [4], 'wide': True, 'default': 0, 'train_dtype': torch.float},
    'ranges': {'dtype': np.int32, 'dim': [2], 'wide': True, 'default': 0, 'train_dtype': torch.int64},
    'ordinals': {'dtype': np.int32, 'dim': [], 'wide': True, 'default': -1, 'train_dtype': torch.float},
    'cardinality': {'dtype': np.int32, 'dim': [], 'wide': False, 'default': 0, 'train_dtype': torch.float},
    'org_bboxes': {'dtype': np.uint16, 'dim': [4]},
    'ocr_ranges': {'dtype': np.int32, 'dim': [2]},
    'token_map': {'dtype': np.int16, 'dim': [], 'wide': True, 'default': -1, 'train_dtype': torch.int64},
    'masks': {'dtype': np.bool, 'dim': [], 'wide': True, 'default': 0, 'train_dtype': torch.uint8},
    'masked_word_ids': {'dtype': np.int32, 'dim': [], 'wide': False, 'default': -1, 'train_dtype': torch.int64},
    'token_label_ids': {'dtype': np.int16, 'dim': [], 'wide': True, 'default': -1, 'train_dtype': torch.int64},
    'doc_id': {'dtype': 'U100', 'dim': [], 'wide': False, 'default': ''},
    'label_name': {'dtype': 'U100', 'dim': [], 'wide': False, 'default': ''},
    'img_lst': {'dtype': np.uint8, 'train_dtype': torch.float},
    'path': {'dtype': 'U256', 'dim': [], 'wide': False, 'default': ''},
}


def get_bpe_positions(token_pos: Tuple[int, int, int, int], bpe_lens: Sequence[int]) -> Sequence[Sequence[float]]:
    pos_lst = []
    tok_len = max(1, sum(bpe_lens))
    offset = 0
    x1, y1, x2, y2 = token_pos
    xwidth = x2 - x1
    assert xwidth >= 0, f'not correct token postions: {token_pos}'
    for bpe_len in bpe_lens:
        pos_lst.append([x1 + (offset / tok_len) * xwidth, y1, x1 + ((offset + bpe_len) / tok_len) * xwidth, y2])
        offset += bpe_len
    return pos_lst


def get_data_part(
    from_range: int,
    to_range: int,
    max_bpe: int,
    bpe_tokens: List[str],
    org_tokens_idx: List[int],
    token_label_idx: List[int],
    seg_data: Dict[str, Any],
) -> Tuple[List[str], List[int], List[int], Dict[str, Any]]:
    assert to_range <= len(bpe_tokens)
    part_bpe_tokens = bpe_tokens[from_range:to_range]
    part_org_tokens_idx = org_tokens_idx[from_range:to_range]
    part_token_label_idx = token_label_idx[from_range:to_range]
    # keep only those segments which contains choosen tokens
    part_seg_data: Dict[str, Any] = {}
    for segkey, seg in seg_data.items():
        part_seg_data[segkey] = {}
        if segkey == 'tokens':
            part_seg_data[segkey]['bboxes'] = seg['bboxes'][from_range:to_range]
            part_seg_data[segkey]['org_bboxes'] = seg['org_bboxes'][from_range:to_range]
        elif segkey in ("lines", "pages"):
            part_seg_idx = [max(from_range, rng[0]) < min(to_range, rng[1]) for rng in seg['ranges']]
            for el_key, el_data in seg.items():
                if el_key == 'ranges':
                    part_seg_data[segkey][el_key] = np.clip(el_data[part_seg_idx, :] - from_range, 0, max_bpe + 1)
                elif el_key in ('bboxes', 'org_bboxes', 'ordinals', 'ocr_ranges'):
                    part_seg_data[segkey][el_key] = el_data[part_seg_idx]
                elif el_key == 'cardinality':
                    part_seg_data[segkey][el_key] = el_data
                else:
                    raise ValueError(
                        f"Key {el_key} in seg_data dictionary is not supported " "by get_data_part function"
                    )
    if "images" in seg_data:
        page_idx = part_seg_data["pages"]["ordinals"]
        if page_idx.size == 0:
            page_idx = [0]
        assert len(page_idx) <= 1, "Images are supported only in single-page spliting strategies"
        part_seg_data["images"]["img_data"] = seg_data["images"]["img_data"][page_idx[0]]
        part_seg_data["images"]["img_size"] = seg_data["images"]["img_size"][page_idx[0]]
    if "lazyimages" in seg_data:
        part_seg_data["lazyimages"]["path"] = seg_data["lazyimages"]["path"]

    return part_bpe_tokens, part_org_tokens_idx, part_token_label_idx, part_seg_data


def convert_to_np(data: Sequence[Any], el_name: str) -> np.ndarray:
    ft = FEAT_META[el_name]
    dtype = ft["dtype"]
    dim = ft["dim"]
    if len(data):
        return np.array(data, dtype=dtype)
    else:
        return np.empty((0,) + tuple(dim), dtype=dtype)


def apply_on_nested_dict(fn: Callable, ndict: Dict[str, Any]) -> Dict[str, Any]:
    new_dict: Dict[str, Any] = {}
    for k, v in ndict.items():
        if isinstance(v, dict):
            new_dict[k] = apply_on_nested_dict(fn, v)
        elif v is None:
            new_dict[k] = None
        else:
            new_dict[k] = fn(v, k)
    return new_dict


def add_missing_tokens(orig_lines: np.ndarray, added_elements: np.ndarray, reorder_idx: np.ndarray = None):
    """
    :param orig_lines: numpy array with original lines element to be modified with new values
    :param added_elements: numpy array with additional elements to be added to lines
    :param reorder_idx: numpy 1D array,
        define how to reorder items created by adding additional elements
    :return: numpy array with additional elements,
        numpy 1D array which was used for sorting final array
    """
    # add ranges with missing tokens
    all_ranges = np.vstack((orig_lines, added_elements))
    if reorder_idx is None:
        reorder_idx = np.argsort(all_ranges[:, 0])
    sorted_ranges = all_ranges[reorder_idx]
    return sorted_ranges, reorder_idx


def fix_missing_tokens_in_lines(doc: Doc2d):
    """
    :param doc: Doc2d instance to be fixed
    :return: fixed Doc2d instance with additional elements in seg_data['lines']
    """
    assert 'tokens' in doc.seg_data, "Tokens data need to be present in doc2d to fix missing lines"
    if 'lines' not in doc.seg_data:
        return doc

    # get missing token indexes
    missing_tokens = []
    offset = 0
    for ln_range in doc.seg_data['lines']['ranges']:
        assert ln_range[0] >= offset, f'Ranges need to be in ascending order: {ln_range[0]} >= {offset}'
        if ln_range[0] > offset:
            missing_tokens.extend(list(range(offset, ln_range[0])))
        offset = ln_range[1]
    if offset < len(doc.tokens):
        missing_tokens.extend(list(range(offset, len(doc.tokens))))

    if not missing_tokens:
        return doc

    doc_fixed = deepcopy(doc)

    # generate ranges for missing tokens
    add_ranges = np.hstack((np.array(missing_tokens)[:, None], np.array(missing_tokens)[:, None] + 1))
    doc_fixed.seg_data['lines']['ranges'], reorder_idx = add_missing_tokens(doc.seg_data['lines']['ranges'], add_ranges)

    # add bboxes
    add_boxes = doc.seg_data['tokens']['org_bboxes'][missing_tokens]
    doc_fixed.seg_data['lines']['org_bboxes'], _ = add_missing_tokens(
        doc.seg_data['lines']['org_bboxes'], add_boxes, reorder_idx
    )

    # add ocr_ranges
    if 'ocr_ranges' in doc.seg_data['lines']:
        assert doc.token_ocr_ranges is not None, "Token ocr ranges are required " "to compute missing lines"
        add_ocr_ranges = doc.token_ocr_ranges[missing_tokens]
        doc_fixed.seg_data['lines']['ocr_ranges'], _ = add_missing_tokens(
            doc.seg_data['lines']['ocr_ranges'], add_ocr_ranges, reorder_idx
        )

    return doc_fixed


def single_line_spans(spans_ranges, seg_data):
    """
    Function is truncating spans to be exactly inside one line, this is required
    becouse even noise span sentinels are required to have bbox and such bbox
    is easier to get if all tokens in the span are in the same line
    :param spans_ranges: proposition of noise-span ranges
    :param seg_data: dict of seg_data
    :return: adjusted span ranges
    """

    def _common_part(span, line_span):
        """
        :param span: range of the noise span
        :param line_span: range of the line
        :return: overlap length, overlap range,
            boolean indicating whether span ended in the current line
        """
        overlap = (max(span[0], line_span[0]), min(span[1], line_span[1]))
        return max(0, overlap[1] - overlap[0]), overlap, line_span[1] >= span[1]

    if "lines" not in seg_data:
        return spans_ranges

    new_spans = []
    last_ln_idx = 0
    lines_ranges = seg_data["lines"]["ranges"]
    for span in spans_ranges:
        max_overlap = 0
        best_overlap = None
        for ln_idx in range(last_ln_idx, len(lines_ranges)):
            line_span = lines_ranges[ln_idx]
            # ensure that the last noise span is in the last line
            is_last_line = ln_idx == len(lines_ranges) - 1
            overlap_length, overlap, ended = _common_part(span, line_span)
            if overlap_length > max_overlap or (is_last_line and overlap_length > 0):
                best_overlap = overlap
                max_overlap = overlap_length
            if ended:
                new_spans.append(best_overlap)
                last_ln_idx = ln_idx
                break
    return np.array(new_spans)
