from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Set

from benchmarker.data.document import Doc2d


@dataclass
class Document:
    identifier: str
    document_2d: Doc2d
    annotations: Dict[str, List[str]]


@dataclass
class DataInstance:
    identifier: str
    input_prefix: str
    document_2d: Doc2d
    output_prefix: str
    output: str


class Dataset(ABC):
    """Extract data instances from dataset.

    :param dataset: Dataset to build DataInstances on
    :return: iterator over Documents
    """
    @abstractmethod
    def __iter__(self) -> Iterator[Document]:
        pass

    @staticmethod
    def escape(value: str) -> str:
        """Escape string (e.g., replace spaces with _).

        :param value: string to escape
        :return: escaped string
        """
        return value

    @staticmethod
    def unescape(value: str) -> str:
        """Unescape string (e.g., replace _ with spaces).

        :param value: string to unescape
        :return: unescaped string
        """
        return value

    def output_prefix(self, value: str) -> str:
        """Format key as output_prefix (e.g, append "=").

        This value is prepended to model output before submitting to geval.
        (Model is taught to guess value without this prefix, we add it manually.)

        :param value: key
        :return: modified key
        """
        return value

    @property
    def labels(self) -> Set[str]:
        """Get a complete list of supported labels.

        :return: set of labels
        """
        raise ValueError('Dataset has to provide labels property to return None answers')
