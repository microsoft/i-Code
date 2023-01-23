import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from benchmarker.data.document import Doc2d
from benchmarker.data.utils import convert_to_np
from benchmarker.input_loader.data_loader import DataLoader


class CommonFormatLoader(DataLoader[Union[str, Path]]):
    def process(self, doc: Union[str, Path], **kwargs) -> Doc2d:
        with open(doc, 'r') as inp:
            js = json.load(inp)
            return self.to_doc2d(js)

    def to_doc2d(self, cf: Dict):
        docid = cf['doc_id']
        if list(set(cf['tokens'])) == [' ']:
            cf['tokens'] = []
            cf['positions'] = []
            cf['scores'] = []
            cf['structures']['pages']['structure_value'] = [[0, 0]]
            cf['structures']['lines']['structure_value'] = []
            cf['structures']['lines']['positions'] = []

        tokens = cf['tokens']
        if len(tokens) == 0:
            logging.warning(f'Doc "{docid}" contains no tokens')

        # add visual data
        seg_data: Dict[str, Any] = {}
        if self._toklevel:
            seg_data['tokens'] = {}
            bb = cf['positions']
            bb_arr = np.array(bb, dtype=np.int) if len(bb) else np.empty((0, 4), dtype=np.int)
            seg_data['tokens']['org_bboxes'] = bb_arr

        for level in self._segment_levels_cleaned:
            seg_data[level] = {}
            rng = cf['structures'][level]['structure_value']
            seg_data[level]['ranges'] = convert_to_np(rng, 'ranges')
            bb = cf['structures'][level]['positions']
            seg_data[level]['org_bboxes'] = convert_to_np(bb, 'org_bboxes')
            assert len(bb) == len(rng), "Number of positions does not match " "number of token ranges"

        return Doc2d(tokens=tokens, seg_data=seg_data, docid=docid)
