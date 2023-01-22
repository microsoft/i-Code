from typing import List


def concat(values: List[str], values_separator: str) -> List[str]:
    return [f' {values_separator} '.join(values)]


def first_item(values: List[str], *_) -> List[str]:
    return [values[0]]


def all_items(values: List[str], *_) -> List[str]:
    return values


def shortest(values: List[str], *_) -> List[str]:
    return [min(values, key=len)]


def longest(values: List[str], *_) -> List[str]:
    return [max(values, key=len)]
