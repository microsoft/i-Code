from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UdopModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from."""

    model_name_or_path: str = field(
        metadata={
            'help':
            'Path to pretrained model or model identifier from huggingface.co/models'
        })
    backbone_weights_prefix: str = field(
        default=None,
        metadata={'help': 'Prefix of path to visual backbone weights'})
    config_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained config name or path if not the same as model_name'
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained tokenizer name or path if not the same as model_name'
        })
    use_fast: bool = field(
        default=False,
        metadata={'help': 'Set this flag to use fast tokenization.'})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Where do you want to store the pretrained models downloaded from s3'
        })
    attention_type: str = field(
        default="block_sparse",
        metadata={"help": "Attention type: BigBird configuruation only. Choices: block_sparse (default) or original_full"},
    )
    attention_window: int = field(
        default=512,
        metadata={"help": "Attention window for Bigbird or Longformer: default is 512"},
    )
    num_attention_heads: int = field(
        default=None,
        metadata={"help": "Num attention head"},
    )
    num_attention_seq_chunk: int = field(
        default=1
    )
    embedding_2dpos_type: str = field(
        default="embed",
        metadata={"help": "embed,sin,none"},
    )
    embedding_use_mlp: bool = field(
        default=False
    )
    embedding_skip_connect: bool = field(
        default=False
    )
    embedding_1dpos_type: str = field(
        default="embed",
        metadata={"help": "same as embedding_2dpos_type"},
    )
    embedding_use_hw: bool = field(
        default=False
    )


@dataclass
class MultifunModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from."""

    model_name_or_path: str = field(
        metadata={
            'help':
            'Path to pretrained model or model identifier from huggingface.co/models'
        })
    model_type: str = field(
        default=None, metadata={'help': 'Model type selected in the list.'})
    config_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained config name or path if not the same as model_name'
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained tokenizer name or path if not the same as model_name'
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Where do you want to store the pretrained models downloaded from huggingface.co'
        },
    )
    model_revision: str = field(
        default='main',
        metadata={
            'help':
            'The specific model version to use (can be a branch name, tag name or commit id).'
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            'help':
            'Will use the token generated when running `transformers-cli login` (necessary to use this script '
            'with private models).'
        },
    )
