from dataclasses import dataclass, field


@dataclass
class UdopDataArguments:
    """Arguments pertaining to what data we are going to input our model for
    training and eval."""

    model_type: str = field(
        default=None, metadata={'help': 'Model type selected in the list.'})
    data_dir: str = field(
        default=None,
        metadata={
            'help':
            'The input data dir. Should contain the .json files for the SQuAD task.'
        },
    )
    train_file: str = field(
        default=None,
        metadata={
            'help': 'Filename of the .json file for the SQuAD task train set.'
        },
    )
    dev_file: str = field(
        default=None,
        metadata={
            'help': 'Filename of the .json file for the SQuAD task dev set.'
        },
    )
    test_file: str = field(
        default=None,
        metadata={
            'help': 'Filename of the .json file for the SQuAD task test set.'
        },
    )
    ocr_dir: str = field(default='cdip-images-full-clean-ocr021121')
    img_dir: str = field(default='cdip-images')
    img_ext: str = field(default='.tiff')
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help':
            'Whether to pad all samples to model maximum sentence length. '
            'If False, will pad the samples dynamically when batching to the maximum length in the batch. More '
            'efficient on GPU but very bad for TPU.'
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )
    max_seq_length_decoder: int = field(
        default=512,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization for decoder. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )    
    mlm_probability: float = field(
        default=0.15,
        metadata={'help': 'token masking probability in mlm'},
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={'help': 'mean_noise_span_length in t5 mlm'},
    )    
    img_drop_rate: float = field(
        default=0.3,
        metadata={'help': 'ocr line masking probability in image masking'},
    )
    whole_word_masking: bool = field(
        default=False,
        metadata={'help': 'Whether to do whole word masking in mlm'})
    mask_mlm_tokens: bool = field(
        default=True,
        metadata={
            'help':
            'Whether to mask image area corresponding to tokens masked in mlm'
        },
    )
    mlm_tokens_mask_method: str = field(
        default='black',
        metadata={
            'help': 'chosen in `["black", "random_grey"]`, default: `"black"`'
        },
    )
    image_drop_rate: float = field(
        default=0.75,
        metadata={'help': 'randomly drop out image'},
    )
    dataset_size: int = field(default=-1, )
    overwrite_cache: bool = field(
        default=False,
        metadata={'help': 'Overwrite the cached training and evaluation sets'},
    )
    max_position_embeddings:int = field(
        default=512
    )
    image_size:int = field(
        default=224
    )
    use_line_tokens: bool = field(
        default=False
    )
    cache_blob_json: bool = field(
        default=False
    )
    do_supervised: bool = field(
        default=False,
        metadata={
            'help': 'do supervised tasks on lang'
        },
    )
    do_selfsupervised: bool = field(
        default=False,
        metadata={
            'help': 'do self supervised tasks on lang'
        },
    )
    do_selfsupervised_vis: bool = field(
        default=False,
        metadata={
            'help': 'do self supervised vision tasks'
        },
    )
    publaynet_dir: str = field(
        default='publaynet',
        metadata={
            'help': 'publaynet directory'
        },
    )
    websrc_dir: str = field(
        default='websrc',
        metadata={
            'help': 'websrc directory'
        },
    )
    docbank_dir: str = field(
        default='docbank',
        metadata={
            'help': 'docbank directory'
        },
    )
    visualmrc_dir: str = field(
        default='visualmrc',
        metadata={
        },
    )
    duebenchmark_dir: str = field(
        default='due_benchmark',
        metadata={
            'help': 'duebenchmark directory'
        },
    )
    rvlcdip_dir: str = field(
        default='rvl-cdip',
        metadata={
            'help': 'rvlcdip directory'
        },
    )
    mpdfs_dir: str = field(
        default='.',
        metadata={
            'help': 'mpdfs directory'
        },
    )
    mae_version: str = field(
        default='.',
        metadata={
            'help': 'mae version'
        },
    )
    mae_checkpoint: str = field(
        default='.',
        metadata={
            'help': 'mae checkpoint'
        },
    )



