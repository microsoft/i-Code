import random
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Iterator, Optional

from benchmarker.data.reader.benchmark_dataset import BenchmarkCorpusMixin
from benchmarker.data.reader.common import DataInstance, Dataset, Document
from benchmarker.data.reader.qa_strategies import concat


class Corpus(BenchmarkCorpusMixin):
    def __init__(
        self,
        train: Optional[Dataset] = None,
        dev: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        unescape_prefix: bool = True,
        unescape_values: bool = True,
        use_prefix: bool = True,
        prefix_separator: str = ':',
        values_separator: str = '|',
        single_property: bool = True,
        use_none_answers: bool = False,
        case_augmentation: bool = False,
        lowercase_expected: bool = False,
        lowercase_input: bool = False,
        train_strategy: Callable = concat,
        dev_strategy: Callable = concat,
        test_strategy: Callable = concat,
        augment_tokens_from_file: Optional[str] = None,
    ):
        """Stores references to dev, train and test Datasets and produces
        data instances on the fly, assuming the configuration provided.

        :param train: Dataset with training instances
        :param dev: Dataset with develop instances
        :param test: Dataset with test set instances
        :param unescape_prefix: whether unescape prefix (e.g., replace _ with spaces)
        :param unescape_values: whether unescape values (e.g., replace _ with spaces)
        :param use_prefix: if set to True, input document will be prefixed with property name
        :param prefix_separator: string to place between property name and input text
        :param values_separator: string to place between values in a case of multi-value properties
        :param single_property: whether assume single-property inference (see Multi-property extraction paper)
        :param case_augmentation: whether to case-augment training documents
        :param lowercase_expected: whether to lowercase expected system output
        :param lowercase_input: lowercase input document (including prefix)
        :param train_strategy: chooses values from trainset
        :param dev_strategy: chooses values from devset
        :param testset_strategy: chooses values from testset
        :param augment_tokens_from_file: path to synonyms dictionary
        """
        self._train: Dataset = train
        self._test: Dataset = test
        self._dev: Dataset = dev

        self._single_property = single_property
        self._unescape_prefix = unescape_prefix
        self._unescape_values = unescape_values
        self._use_prefix = use_prefix
        self._prefix_separator = prefix_separator
        self._values_separator = values_separator
        self._single_property = single_property
        self._use_none_answers = use_none_answers
        self._case_augmentation = case_augmentation
        self._lowercase_expected = lowercase_expected
        self._lowercase_input = lowercase_input
        self._train_strategy = train_strategy
        self._dev_strategy = dev_strategy
        self._test_strategy = test_strategy

        self._paraphrases = None

        self._validate_config()
        self._prepare_augmenter(augment_tokens_from_file)

    def _prepare_augmenter(self, augment_tokens_from_file: Optional[str] = None):
        self._aug_dict = None
        self._aug_counter = 0
        if augment_tokens_from_file:
            self._aug_dict = defaultdict(set)
            with open(augment_tokens_from_file) as ins:
                for line in ins:
                    tokens = [t.lower() for t in line.rstrip().split()]
                    for tok in tokens:
                        self._aug_dict[tok].update([t.replace('_', ' ') for t in tokens])

    def _augment(self, token: str):
        if not self._aug_dict or token.lower() not in self._aug_dict:
            return token
        candidates = list(self._aug_dict[token.lower()])
        random.shuffle(candidates)
        return candidates[0]

    def _validate_config(self):
        assert not (self._lowercase_input and self._case_augmentation), 'Do not use lowercasing with case augmentation'
        assert self._single_property, 'Multi-property is not supported yet'

    def doc_to_instances(
        self, document: Document, dataset: Dataset, strategy: Callable
    ) -> Optional[Iterator[DataInstance]]:
        """Extract training instances from document.

        :param document: Document obtained from reader
        :param dataset: Dataset the document was sourced from
        :return: iterator over DataInstances
        """
        assert self._single_property
        if not dataset:
            return None

        keys = dataset.labels if self._use_none_answers else document.annotations.keys()

        for key in keys:
            values = document.annotations[key]
            if not values:
                values = ['None']

            if self._use_prefix:
                prefix = self._paraphrases[key] if self._paraphrases else key
                prefix = f'{dataset.unescape(prefix) if self._unescape_prefix else prefix} {self._prefix_separator} '
            else:
                prefix = ''

            if self._unescape_values:
                values = [dataset.unescape(v) for v in values]

            document.document_2d.tokens = [self._augment(t) for t in document.document_2d.tokens]

            if self._lowercase_input:
                document.document_2d.tokens = [t.lower() for t in document.document_2d.tokens]
                prefix = prefix.lower()

            output_prefix = dataset.output_prefix(key)

            for value in strategy(values, self._values_separator):

                if self._lowercase_expected:
                    value = value.lower()

                yield DataInstance(document.identifier, prefix, document.document_2d, output_prefix, value)

    def get_instances(
        self, dataset: Dataset, strategy: Callable, case_augmentation=False
    ) -> Optional[Iterator[DataInstance]]:
        """Extract data instances from dataset.

        :param dataset: Dataset to build DataInstances on
        :param case_augmentation: bool indicating if document should be case augmented
        :return: iterator over DataInstances

        """

        def generator():
            # Do not touch this unless you know what it is doing
            for doc in dataset:
                if case_augmentation:
                    yield from (
                        ins for doc in case_augmenter(doc) for ins in self.doc_to_instances(doc, dataset, strategy)
                    )
                else:
                    yield from self.doc_to_instances(doc, dataset, strategy)

        if dataset is not None:
            return generator()
        return None

    @property
    def train(self) -> Optional[Iterator[DataInstance]]:
        """Train set DataInstances."""
        return self.get_instances(self._train, self._train_strategy, case_augmentation=self._case_augmentation)

    @property
    def dev(self) -> Optional[Iterator[DataInstance]]:
        """Dev set DataInstances."""
        return self.get_instances(self._dev, self._dev_strategy)

    @property
    def test(self) -> Optional[Iterator[DataInstance]]:
        """Test set DataInstances."""
        return self.get_instances(self._test, self._test_strategy)


def case_augmenter(doc: Document):
    """
    :param doc: Document which will be augmented with different casing

    """
    # yield original doc first
    yield doc
    check_limit = 100  # for faster checking of document casing
    # iterate over lower and upper func
    for func in (lambda x: x.lower(), lambda x: x.upper()):
        tokens = [func(tok) for tok in doc.document_2d.tokens]
        # skip instance if original document is already uppercased or lowercased
        if tokens[:check_limit] == doc.document_2d.tokens[:check_limit]:
            continue
        new_doc = deepcopy(doc)
        # change annotations as well, skip augmenting for None values
        new_doc.annotations = defaultdict(
            list,
            {k: [func(item) if item.lower() != "none" else item for item in v] for k, v in doc.annotations.items()},
        )
        new_doc.document_2d.tokens = tokens
        yield new_doc
