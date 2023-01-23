from abc import ABCMeta, abstractmethod
from collections.abc import Iterator as IterBaseClass
from typing import Generic, Iterable, Iterator, Optional, Sequence, TypeVar

from benchmarker.data.document import Doc2d

T = TypeVar('T')


class DataLoader(IterBaseClass, Generic[T], metaclass=ABCMeta):

    def __init__(self, docs: Iterable[T], segment_levels: Optional[Sequence[str]] = None) -> None:
        # pages and tokens are mandatory segment_levels which are computed for each doc2d
        self._mandatory_levels: Sequence[str] = ('tokens', 'pages')
        self._segment_levels: Sequence[str] = tuple(set((list(segment_levels) if segment_levels
                                                      else []) + list(self._mandatory_levels)))
        self._segment_levels_cleaned = tuple([s for s in self._segment_levels
                                              if s in ("lines", "pages")])
        self._toklevel = 'tokens' in self._segment_levels

        self.inputs = iter(docs)

    @property
    def segment_levels(self) -> Sequence[str]:
        """Segment levels, it could be one of:
            * 'tokens'
            * 'lines'
            * 'pages'
        """
        return self._segment_levels

    def __next__(self) -> Doc2d:
        return self.process(next(self.inputs))

    def __iter__(self) -> Iterator[Doc2d]:
        return self

    @abstractmethod
    def process(self, doc: T, **kwargs) -> Doc2d:
        pass
