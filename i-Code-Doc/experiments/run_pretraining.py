import argparse
import json
import logging
import os
import sys
from pathlib import Path

import transformers
from transformers import HfArgumentParser
from transformers.trainer import Trainer
from transformers.trainer_utils import is_main_process, set_seed

from core.datasets import (MultilingualPdfDataset, MultilingualPdfDataTrainingArguments,
                           RVLCdipImageDataset, Publaynet, Docbank, Websrc, VisualMRC, Duebenchmark)
from torch.utils.data.dataset import ConcatDataset

from core.models import (UDOPDualForConditionalGeneration,
                         UDOPUnimodelForConditionalGeneration,
                         UDOPDualConfig,
                         UDOPUnimodelConfig,
                         UDOPTokenizer,
                         )

from core.trainers import (OneocrModelArguments, PretrainTrainer, OneocrTrainingArguments)
from transformers.training_args import TrainingArguments
from core.common.utils import get_last_checkpoint
from core.trainers.data_collator import PretrainingDataCollator

import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'UDOP-Unimodel': (UDOPUnimodelConfig, UDOPUnimodelForConditionalGeneration, UDOPTokenizer),
    'UDOP-Dual': (UDOPDualConfig, UDOPDualForConditionalGeneration, UDOPTokenizer),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((
        OneocrModelArguments,
        MultilingualPdfDataTrainingArguments,
        TrainingArguments,
    ))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # model_args, data_args, training_args = parser.parse_json_file(
        #     json_file=os.path.abspath(sys.argv[1]))
        args_dict = json.loads(os.path.abspath(sys.argv[1])).read_text()
        # args_dict['local_rank'] = args.local_rank
        model_args, data_args, training_args = parser.parse_dict(args_dict)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.local_rank = training_args.local_rank
    data_args.local_rank = training_args.local_rank
    training_args.visualize_mae = data_args.visualize_mae
    training_args.alternate_steps = data_args.alternate_steps
    #training_args.local_rank = args.local_rank

    model_args.seed = training_args.seed
    data_args.seed = training_args.seed
    data_args.padding = "max_length" if data_args.pad_to_max_length else "longest"
    set_seed(training_args.seed)

    training_args.logging_dir = os.path.join(training_args.output_dir, 'runs')
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(training_args.output_dir, 'cache')
    os.makedirs(model_args.cache_dir, exist_ok=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.')
        elif last_checkpoint is not None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        +
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info('Training/evaluation parameters %s', training_args)

    if data_args.model_type in MODEL_CLASSES:
        config_type, model_type, tokenizer_type = MODEL_CLASSES[data_args.model_type]
    else:
        # config_type, model_type, tokenizer_type = AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
        config_type, model_type, tokenizer_type = None, None, None

    if 'checkpoint' in model_args.model_name_or_path:
        model_args.model_name_or_path = os.path.join(data_args.data_dir, model_args.model_name_or_path)

    config = config_type.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        backbone_weights_prefix=model_args.backbone_weights_prefix,
        max_seq_len=data_args.max_seq_length,
    )
    config.data_dir = data_args.data_dir
    config.mae_version = data_args.mae_version
    config.mae_checkpoint = data_args.mae_checkpoint
    
    config.embedding_1dpos_type = model_args.embedding_1dpos_type
    config.embedding_2dpos_type = model_args.embedding_2dpos_type
    config.embedding_use_mlp = model_args.embedding_use_mlp
    config.embedding_skip_connect = model_args.embedding_skip_connect
    config.embedding_use_hw = model_args.embedding_use_hw

    config.image_size = data_args.image_size
    #if training_args.fp16:
    if training_args.fp16_backend == 'apex':
        config.layernorm_type = 'apex_FusedLayerNorm'
        config.syncbn_type = 'apex_SyncBatchNorm'
    else:
        config.layernorm_type = 'torch_LayerNorm'
        config.syncbn_type = 'torch_SyncBatchNorm'

            
    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool('.ckpt' in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=True,
    )
    tokenizer = tokenizer_type.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        add_prefix_space=data_args.add_prefix_space,
        use_fast=True,
    )
    if data_args.max_seq_length > tokenizer.model_max_length:
        tokenizer.model_max_length = data_args.max_seq_length

    model.resize_token_embeddings(len(tokenizer))
    data_collator_class = PretrainingDataCollator
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)
    
    if training_args.do_train:
        train_dataset = {}
        if data_args.do_selfsupervised or data_args.do_supervised:
            train_dataset['lang'] = []
        if data_args.do_selfsupervised_vis:
            train_dataset['vis'] = []
        
        if data_args.do_selfsupervised:
            train_dataset['lang'] += [MultilingualPdfDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, task='text2bbox'), MultilingualPdfDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, task='bbox2text'), MultilingualPdfDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, task='ocr')]
            
        if data_args.do_supervised:
            train_dataset['lang'] += [Publaynet(data_args, tokenizer=tokenizer, task='layout'), Docbank(data_args, tokenizer=tokenizer, task='layout'), Docbank(data_args, tokenizer=tokenizer, task='ie'), Websrc(data_args, tokenizer=tokenizer, task='qa'), VisualMRC(data_args, tokenizer=tokenizer, task='qa'), Duebenchmark(data_args, tokenizer=tokenizer)]

        if train_dataset == {}:
            train_dataset['lang'] = [MultilingualPdfDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, task='mlm'),]
            
        for key in train_dataset:
            train_dataset[key] = ConcatDataset(train_dataset[key])
    # Get datasets

    eval_dataset = (MultilingualPdfDataset(
        data_args,
        tokenizer=tokenizer,
        mode='dev',
        cache_dir=model_args.cache_dir,
    ) if training_args.do_eval else None)
    
    print(f"Using {training_args.half_precision_backend} half precision backend")
    
    # Initialize our Trainer
    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_class(
            tokenizer=tokenizer,
            padding=data_args.padding,
            max_length=data_args.max_seq_length,
            max_length_decoder=data_args.max_seq_length_decoder,
            #local_rank=training_args.local_rank
            #pad_to_multiple_of=None,
        )
    )
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        #elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        #    checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, 'trainer_state.json'))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, 'eval_results.txt')
        if trainer.is_world_process_zero():
            with open(output_eval_file, 'w', encoding='utf8') as writer:
                logger.info('***** Eval results *****')
                for key, value in result.items():
                    logger.info('    %s = %s', key, value)
                    writer.write(f'{key} = {value}\n')

            results.update(result)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == '__main__':
    main()
