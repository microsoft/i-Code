from typing import Any, Dict, Optional, Sequence

from core.models import UdopConfig


class BaseBenchmarkerConfig(UdopConfig):
    """Configuration class to store config of BENCHMARKER models.

    Attributes:
        model_type(str): Type of BENCHMARKER's base model. Supported types: {SUPPORTED_MODEL_TYPES}
        max_context_weight(float): target value of context weight (Default = 1.0)
        context_weight_update_time(float): fraction of total training time to increase
         context weight for (Default: 0.5)
        max_pos_dropout(float): target value of positional dropout (Default = 0.0)
        pos_dropout_update_time(float): fraction of total training time to increase
         positional dropout for (Default: 0.5)
        positional_dropout_type(str): type of positional dropout to use (Default = 'random')
        training_tasks(Sequence[str]): Types of training task to use. Valid are 'lm'
         (Default = ('lm', ))
        context_embeddings(Sequence[Dict[str, Any]]): List of configurations for context embeddings
         (Default = [])
        relative_bias_args(Sequence[Dict[str, Any]]): List of configurations for relative biases
         (Default = [])
        vision_augmentation(Sequence[Dict[str, Any]]): List of configurations for computer vision
        specific augmentation purposes (Default = [])
        page_embeddings_type(str): type of page embeddings to use. Valid values are 'none' and
         'basic' (Default = 'none')
        page_embeddings_args(Dict[str, Any]): Configuration for page embeddings (Default = None)
        do_lower_case(bool): if true, model uses lower cased tokenizer (Default = False)
    """

    def __init__(self,
                 schema_version: Optional[str] = None,
                 num_doc_classes: int = 40,
                 max_context_weight: float = 1.0,
                 context_weight_update_time: float = 0.5,
                 max_pos_dropout: float = 0.0,
                 pos_dropout_update_time: float = 0.5,
                 positional_dropout_type: str = 'random',
                 training_tasks: Sequence[str] = ('lm', ),
                 context_embeddings: Optional[Sequence[Dict[str, Any]]] = None,
                 relative_bias_args: Optional[Sequence[Dict[str, Any]]] = None,
                 vision_augmentation: Optional[Sequence[Dict[str, Any]]] = None,
                 page_embeddings_type: str = 'none',
                 page_embeddings_args: Optional[Dict[str, Any]] = None,
                 do_lower_case: bool = False,
                 disable_sequential_embeddings: bool = False,
                 word_dropout: float = 0.0,
                 locked_dropout: float = 0.0,
                 attention_dropout: Optional[float] = None,
                 context_residual: Optional[str] = None,
                 truncate_decoder_after_layer: Optional[int] = None,
                 truncate_encoder_after_layer: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.schema_version = schema_version
        self.num_doc_classes = num_doc_classes
        self.max_context_weight = max_context_weight
        self.context_weight_update_time = context_weight_update_time
        self.max_pos_dropout = max_pos_dropout
        self.pos_dropout_update_time = pos_dropout_update_time
        self.positional_dropout_type = positional_dropout_type
        self.training_tasks = training_tasks
        self.context_embeddings = [] if context_embeddings is None else context_embeddings
        self.relative_bias_args = [] if relative_bias_args is None else relative_bias_args
        self.vision_augmentation = [] if vision_augmentation is None else vision_augmentation
        self.page_embeddings_type = page_embeddings_type
        self.page_embeddings_args = {} if page_embeddings_args is None else page_embeddings_args
        self.do_lower_case = do_lower_case
        self.disable_sequential_embeddings = disable_sequential_embeddings
        self.word_dropout = word_dropout
        self.locked_dropout = locked_dropout
        self.attention_dropout = attention_dropout
        self.context_residual = context_residual
        self.truncate_decoder_after_layer = truncate_decoder_after_layer
        self.truncate_encoder_after_layer = truncate_encoder_after_layer

class T5BenchmarkerConfig(BaseBenchmarkerConfig, UdopConfig):
    pass
