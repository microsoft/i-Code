import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarker.cli.l5.common.pl_modules.base_lightning_module import BaseLightningModule
from benchmarker.cli.l5.common.utils import freeze_embeds, label_smoothed_nll_loss, lmap, use_task_specific_params
from benchmarker.utils.training import load_2d_model, load_config, load_tokenizer

from core.common.utils import get_visual_bbox

class L5GenerationModule(BaseLightningModule):
    def __init__(self, hparams: argparse.Namespace, val_metrics, **kwargs):
        super().__init__(hparams, num_labels=None, config=False, model=False, **kwargs)

        self.valid_metrics = torch.nn.ModuleDict(val_metrics)

        config_attr_to_override = (
            "relative_bias_args",
            "gradient_checkpointing",
        )

        for p in config_attr_to_override:
            if getattr(hparams, p) is None:
                delattr(hparams, p)
                
        model_name_or_path = hparams.model_name_or_path
        self.config = load_config(
            model_path=Path(model_name_or_path), **vars(hparams)
        )

        if hparams.load_ckpt_weight is not None:
            ckpt = torch.load(hparams.load_ckpt_weight, map_location="cpu")
            ckpt_state_dict = ckpt["state_dict"]
            ckpt_state_dict = {k[6:]: v for k, v in ckpt_state_dict.items() if k.startswith("model.")}
        else:
            ckpt_state_dict = None
            
        self.tokenizer = load_tokenizer(
            Path(model_name_or_path),
            model_type=hparams.model_type,
        )
        self.model = load_2d_model(
            Path(model_name_or_path),
            config=self.config,
            mode='pretraining',
            model_type=hparams.model_type,
            state_dict=ckpt_state_dict,
        )

        use_task_specific_params(self.model, "generation")

        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)

        self.decoder_start_token_id = None  # default to config
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.setup_copying(self.hparams.only_copying_from_input, self.hparams.copying_exceptions)
        self.length_penalty = self.hparams.length_penalty
        
    def setup_copying(self, only_copying_from_input, copying_exceptions):
        self.only_copying_from_input = self.hparams.only_copying_from_input
        if self.only_copying_from_input:
            self.always_allowed = (
                {self.tokenizer.bos_token_id, self.tokenizer.eos_token_id}.union(
                    set(self.tokenizer(copying_exceptions).input_ids)
                )
            ).difference((None,))
            self.always_forbidden = {self.tokenizer.pad_token_id, self.tokenizer.unk_token_id}

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        seg_data = batch["seg_data"]['tokens']['bboxes']

        tgt_ids = batch["labels"]
        decoder_input_ids = self.model._shift_right(tgt_ids)
        
        # self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        image = batch['seg_data']['lazyimages']['img_lst']
        batch_size, _, image_size, _ = image.shape
        visual_seg_data = torch.tensor(get_visual_bbox(image_size).float())[None, :].repeat(batch_size, 1, 1).to(image)
        
        outputs = self(
            src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            seg_data=seg_data,
            visual_seg_data=visual_seg_data,
            image=image,
        )
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        return loss

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {'loss': loss_tensors.mean()}
        lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(self.trainer.optimizers[0].param_groups)}
        logs.update(lrs)
        self.log_dict(logs)
        return loss_tensors.mean()

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def collate_metrics(self, outputs):
        generative_metrics = {
            metric_name: np.array([x[metric_name] for x in outputs]).mean() for metric_name in ["gen_time", "gen_len"]
        }
        return generative_metrics

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        if outputs:
            loss = torch.stack([x['loss'] for x in outputs]).mean()
        else:
            loss = torch.tensor(1000.0)
        generative_metrics = self.collate_metrics(outputs)
        generative_metrics['loss'] = loss.item()
        all_metrics = {f"{prefix}_{k}": x for k, x in generative_metrics.items()}

        self.log_dict(all_metrics)
        self.log(f"{prefix}_loss", loss)
        for metric_name in self.valid_metrics:
            self.log(f"{prefix}_{metric_name}", self.valid_metrics[metric_name])

        flat_generations = [
            dict(zip(out["generation_results"].keys(), v))
            for out in outputs
            for v in zip(*out["generation_results"].values())
        ]
        self.val_outs = {
            "generation_results": flat_generations,
        }
        return {
            "generation_results": flat_generations,
        }

    def collect_predictions(self, data):
        predictions = []
        references = []
        if self.hparams.val_metric[0] == 'anls':
            for (pred, ref) in zip(data['preds'], data['target']):
                ans_items = [pred.strip()]
                ref_items = [i.strip() for i in ref.split('|')]
                predictions.append(ans_items)
                references.append(ref_items)
        else:
            if 'label_name' in data:
                for (label, pred, ref) in zip(data['label_name'], data['preds'], data['target']):
                    if pred != 'None':
                        ans_items = [f'{label}{i.strip()}' for i in pred.split('|')]
                    else:
                        ans_items = []
                    if ref != 'None':
                        ref_items = [f'{label}{i.strip()}' for i in ref.split('|')]
                    else:
                        ref_items = []
                    predictions.append(ans_items)
                    references.append(ref_items)
            else:
                for (pred, ref) in zip(data['preds'], data['target']):
                    ans_items = [i.strip() for i in pred.split('|')]
                    ref_items = [i.strip() for i in ref.split('|')]
                    predictions.append(ans_items)
                    references.append(ref_items)
        return (predictions, references)

    def ids_to_clean_text(self, generated_ids: List[List[int]]) -> List[str]:
        truncaded_ids = []
        for gen_line in generated_ids:
            nz = gen_line.nonzero(as_tuple=False)
            if nz.shape[0] == 0:  # It is possible there are only zeros when decoder has changed
                truncaded_ids.append([])
            else:
                last = nz.max() + 1
                truncaded_ids.append(gen_line[:last])
        gen_text = self.tokenizer.batch_decode(
            truncaded_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return lmap(str.strip, gen_text)

    def _generative_step(self, batch: dict, prefix='dev') -> dict:
        t0 = time.time()

        if self.only_copying_from_input:
            _allowed_tokens = [
                list(set(i).difference(self.always_forbidden).union(self.always_allowed))
                for i in batch["input_ids"].tolist()
            ]

            def prefix_allowed_tokens_fn(batch_id, sent):
                return _allowed_tokens[batch_id]

        else:
            prefix_allowed_tokens_fn = None
        
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        seg_data = batch["seg_data"]['tokens']['bboxes']

        image = batch['seg_data']['lazyimages']['img_lst']
        batch_size, _, image_size, _ = image.shape
        visual_seg_data = torch.tensor(get_visual_bbox(image_size).float())[None, :].repeat(batch_size, 1, 1).to(image)
        
        generated_ids = self.model.generate(
            src_ids,
            attention_mask=src_mask,
            seg_data=seg_data,
            visual_seg_data=visual_seg_data,
            image=image,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            length_penalty=self.length_penalty,
        )
        preds: List[str] = self.ids_to_clean_text(generated_ids)

        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]

        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {'loss': loss_tensors.mean()}
        summ_len = np.mean(lmap(len, generated_ids))
        generation_results = {"preds": preds, "target": target, "is_equal": [p == t for p, t in zip(preds, target)]}
        if "doc_id" in batch:
            generation_results.update(doc_id=batch["doc_id"])
        if "label_name" in batch:
            generation_results.update(label_name=batch["label_name"])

        predictions = self.collect_predictions(generation_results)
        for metric_name in self.valid_metrics:
            self.valid_metrics[metric_name](predictions[0], predictions[1])

        base_metrics.update(gen_time=gen_time, gen_len=summ_len, generation_results=generation_results)

        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch, prefix='test')

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        raise NotImplementedError("dataloader should be assigned from separate DataModule")

    @property
    def dataset_size(self):
        return len(self.train_dataloader().dataset)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseLightningModule.add_model_specific_args(parser, root_dir)
        parser.add_argument("--model_type", type=str, default="t5")
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--overwrite_output_dir", action="store_true", default=False)
        parser.add_argument("--load_ckpt_weight", type=str, default=None)
        parser.add_argument("--restore_training", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument(
            "--logger_name", type=str, choices=["default", "wandb", "wandb_shared", "mlflow"], default="default"
        )

        group = parser.add_argument_group("mlflow")
        group.add_argument("--mlflow_experiment", type=str, default="/trash", help="MLFlow experiment name")
        group.add_argument("--mlflow_uri", type=str, default="http://10.2.1.13:23889/", help="tracking uri")
        group.add_argument("--mlflow_tags", default=None, type=json.loads)

        parser.add_argument("--word_dropout", type=float, default=0.0, required=False)
        parser.add_argument("--locked_dropout", type=float, default=0.0, required=False)

        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)

        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument("--length_penalty", type=float, default=1.0)
        parser.add_argument("--only_copying_from_input", action="store_true", default=False)
        parser.add_argument("--copying_exceptions", type=str, default="yes no Yes No")

        parser.add_argument("--context_residual", type=str, default=None, required=False)

        parser.add_argument("--eval_max_gen_length", type=int, default=None, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, "
            "not epochs. So val_check_interval will effect it.",
        )
        parser.add_argument(
            '--relative_bias_args',
            type=json.loads,
            default='[{"type":"1d"},{"type":"horizontal"},{"type":"vertical"}]',
            help="list of positional biases to use and add to attention matrix",
        )
        parser.add_argument(
            '--context_embeddings',
            type=json.loads,
            default=None,
            help="list of context embeddings to use. Supports vision augmentation too.)",
        )

        parser.add_argument(
            "--truncate_decoder_after_layer",
            type=int,
            default=None,
            help="Overwrite number of decoder layers in pretrained model",
        )
        parser.add_argument(
            "--truncate_encoder_after_layer",
            type=int,
            default=None,
            help="Overwrite number of encoder layers in pretrained model",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            action="store_true",
            default=False,
            help='Use gradient checkpointing (multi-head attention activations are not saved, but recalculated in backward pass).'
            'Reduces memory usage about 2.5 times and slows training by about 15% (still worth it e.g. bigger batch size).',
        )

        return parser
