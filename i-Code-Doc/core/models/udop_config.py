from typing import Any, Dict, Optional, Sequence

from transformers.models.t5.configuration_t5 import T5Config

UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class UdopConfig(T5Config):
    pretrained_config_archive_map = UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, 
                 max_2d_position_embeddings=1024,
                 max_bbox_length=1001, 
                 mae_version = 'mae_vit_large_patch16',
                 mae_checkpoint = 'mae-models/mae_pretrain_vit_large_full.pth',
                 image_size: int = 224,
                 relative_bias_args: Optional[Sequence[Dict[str, Any]]] = [{"type":"1d"},{"type":"horizontal"},{"type":"vertical"}],
                 truncate_decoder_after_layer: Optional[int] = None,
                 truncate_encoder_after_layer: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_bbox_length = max_bbox_length
        self.mae_version = mae_version
        self.mae_checkpoint = mae_checkpoint

        self.relative_bias_args = [] if relative_bias_args is None else relative_bias_args
        self.image_size = image_size
        
        self.truncate_decoder_after_layer = truncate_decoder_after_layer
        self.truncate_encoder_after_layer = truncate_encoder_after_layer
