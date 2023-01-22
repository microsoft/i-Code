from benchmarker.__version__ import __version__

from benchmarker.config.benchmarker_config import (
    T5BenchmarkerConfig,
)
from core.models import (UdopDualForConditionalGeneration,
                         UdopUnimodelForConditionalGeneration,
                         UdopConfig,
                         UdopTokenizer,
                        )

__all__ = ['__version__']

MODEL_CLASSES = {
    "UdopUnimodel": {
        "config": UdopConfig,
        "config2d": UdopConfig,
        "tokenizer": UdopTokenizer,
        "model_attr_name": "UdopUnimodel",
        "wordgap_data_converter": None,
        "pretraining": UdopUnimodelForConditionalGeneration,
        "token_classification": UdopUnimodelForConditionalGeneration,
    },
    "UdopDual": {
        "config": UdopConfig,
        "config2d": UdopConfig,
        "tokenizer": UdopTokenizer,
        "model_attr_name": "UdopDual",
        "wordgap_data_converter": None,
        "pretraining": UdopDualForConditionalGeneration,
        "token_classification": UdopDualForConditionalGeneration,
    },
    
}

MODEL_CLASSES_REVERSE = {
    model_class: (model_type, model_kind)
    for model_type, model_dict in MODEL_CLASSES.items()
    for model_kind, model_class in model_dict.items()
    if isinstance(model_class, type)
}
