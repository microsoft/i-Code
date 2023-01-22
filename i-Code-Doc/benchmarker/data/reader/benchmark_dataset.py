import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Union

from benchmarker.data.reader.common import Dataset, Document
from benchmarker.input_loader.common_format import CommonFormatLoader

logger = logging.getLogger(__name__)


def get_value(annotation_value: Dict) -> List:
    if 'value_variants' in annotation_value:
        return annotation_value['value_variants']
    else:
        return [annotation_value['value']]


def get_child_values(annotation_values: List[Dict]) -> List:
    values: List = []
    for annotation_value in annotation_values:
        values += [annotation_value['value']]

    return values


class BenchmarkDataset(Dataset):
    """docstring for BenchmarkDataset"""

    def __init__(self, directory: Path, split, ocr: str, segment_levels: tuple = ("tokens", "pages")):
        super(BenchmarkDataset, self).__init__()
        self.directory = directory
        self.split = split
        self.ocr = ocr
        self.segment_levels = segment_levels

    def __iter__(self) -> Iterator[Document]:
        docs_jsonl_path = self.directory / self.split / 'document.jsonl'
        docs_content_jsonl_path = self.directory / self.split / 'documents_content.jsonl'
        with open(docs_jsonl_path) as docs_file, open(docs_content_jsonl_path) as docs_content_file:
            for doc_line, doc_content in zip(docs_file, docs_content_file):
                doc_dict = json.loads(doc_line)
                identifier = f'{doc_dict["name"]}'
                doc_content_dict = json.loads(doc_content)
                tool2cf = {c['tool_name']: c for c in doc_content_dict['contents']}
                if self.ocr not in tool2cf:
                    logging.warning(f'No common format for {doc_dict["name"]}. Skipping it')
                    continue
                if not tool2cf[self.ocr]['common_format']['tokens']:
                    logging.warning(f'No tokens in common format for {doc_dict["name"]}. Skipping it')
                    continue
                common_format = tool2cf[self.ocr]['common_format']
                loader = CommonFormatLoader([], segment_levels=self.segment_levels)
                doc2d = loader.to_doc2d(common_format)
                img_dir = self.directory / 'png' / identifier.split('.pdf')[0]
                if not img_dir.exists():
                    logger.warning(f"Cannot locate directory {img_dir}")
                doc2d.seg_data['lazyimages'] = {'path': img_dir}

                for annotation in doc_dict['annotations']:
                    annotations = defaultdict(list)
                    question = annotation['key']

                    values = []
                    for i, value in enumerate(annotation['values']):
                        if 'children' in value:
                            # XXX: this part could be specific to PWC dataset and might need some changes
                            # for different datasets with 'children' keys
                            for child in value['children']:
                                child_question = f"What are the {question} values for the {child['key']} column?"
                                annotations[child_question] += get_child_values(child['values'])
                        else:
                            values += get_value(value)

                    if values:
                        annotations[question] = values

                    document = Document(identifier, doc2d, annotations)
                    yield document

    def output_prefix(self, value: str) -> str:
        """Format key as output_prefix (e.g, append "=").

        This value is prepended to model output before submitting to geval.
        (Model is taught to guess value without this prefix, we add it manually.)

        :param value: key
        :return: modified key
        """

        if os.path.basename(self.directory).startswith('kleister'):
            return f'{value}='
        return value


class BenchmarkCorpusMixin:
    def read_benchmark_challenge(self, directory: Union[str, Path], **kwargs):
        for split in ['train', 'dev', 'test']:
            inner_attribute = '_' + split
            setattr(self, inner_attribute, BenchmarkDataset(directory, split, **kwargs))

