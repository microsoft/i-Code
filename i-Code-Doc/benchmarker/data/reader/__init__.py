from benchmarker.data.reader.corpus import Corpus

__all__ = ['Corpus']

try:
    import tensorflow  # noqa
    import tfds  # noqa
    from benchmarker.data.reader.tensorflow_dataset import TensorFlowDataset

    __all__.append('TensorFlowDataset')
except ImportError:
    pass
