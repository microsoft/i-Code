from typing import Any, Dict, Optional, Sequence, Tuple

from transformers.models.t5.configuration_t5 import T5Config


Baseline_PRETRAINED_MODEL_ARCHIVE_MAP = {}

Baseline_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


CURRENT_SCHEMA_VERSION = 'v1'
SUPPORTED_TRAINING_TASKS = ['lm']
SUPPORTED_DROPOUT_TYPES = ['random', 'tokens', 'features']



class UnimodelConfig(T5Config):
    pretrained_config_archive_map = Baseline_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, 
                 max_2d_position_embeddings=1024,
                 max_bbox_length=1001, 
                 schema_version: Optional[str] = None,
                 num_doc_classes: int = 40,
                 max_context_weight: float = 1.0,
                 context_weight_update_time: float = 0.5,
                 image_size: int = 224,
                 max_pos_dropout: float = 0.0,
                 pos_dropout_update_time: float = 0.5,
                 positional_dropout_type: str = 'random',
                 training_tasks: Sequence[str] = ('lm', ),
                 context_embeddings: Optional[Sequence[Dict[str, Any]]] = None,
                 relative_bias_args: Optional[Sequence[Dict[str, Any]]] = [{"type":"1d"},{"type":"horizontal"},{"type":"vertical"}],
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
        
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_bbox_length = max_bbox_length
        self.florence_version = 'Florence-v1.1-davit-d3'
        self.florence_checkpoint = 'Florence-v1.1/florence-v1.1-davit-d3-224x224-64ccc2.pt'
        self.mae_version = 'mae_vit_large_patch16'
        self.mae_checkpoint = 'mae-models/mae_pretrain_vit_large_full.pth'
        self.data_dir = '.'

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
        self.image_size = image_size
