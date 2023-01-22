from typing import Any, Dict, Sequence


class Span(object):

    def __init__(self, example_id: str, span_index: int, start_position: int, end_position: int,
                 tokens: Sequence[str], masked_positions: Sequence[int],
                 masked_labels: Sequence[str], seg_data: Dict[str, Any],
                 original_tokens_indices: Sequence[int], token_label_indices: Sequence[int]):
        self.__example_id = example_id
        self.__span_index = span_index
        self.__start_position = start_position
        self.__end_position = end_position
        self.__tokens = tokens
        self.__masked_positions = masked_positions
        self.__masked_labels = masked_labels
        self.__seg_data = seg_data
        self.__original_tokens_indices = original_tokens_indices
        self.__token_label_indices = token_label_indices

    @property
    def example_id(self) -> str:
        return self.__example_id

    @property
    def span_index(self) -> int:
        return self.__span_index

    @property
    def start_position(self) -> int:
        return self.__start_position

    @property
    def end_position(self) -> int:
        return self.__end_position

    @property
    def tokens(self) -> Sequence[str]:
        return self.__tokens

    @property
    def masked_positions(self) -> Sequence[int]:
        return self.__masked_positions

    @property
    def masked_labels(self) -> Sequence[str]:
        return self.__masked_labels

    @property
    def seg_data(self) -> Dict[str, Any]:
        return self.__seg_data

    @property
    def original_tokens_indices(self) -> Sequence[int]:
        return self.__original_tokens_indices

    @property
    def token_label_indices(self) -> Sequence[int]:
        return self.__token_label_indices

    def __repr__(self):
        return f'Span[' \
               f'example_id={self.example_id},' \
               f' span_index={self.span_index},' \
               f' start_position={self.start_position}, end_position={self.end_position},' \
               f' tokens={self.tokens}, masked_positions={self.masked_positions},' \
               f' masked_labels={self.masked_labels}, seg_data={self.seg_data},' \
               f' original_tokens_indices={self.original_tokens_indices},' \
               f' token_label_indices={self.token_label_indices}]'
