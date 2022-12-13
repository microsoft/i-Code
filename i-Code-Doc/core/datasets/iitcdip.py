import json
import os
import random
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

from config import UdopDataArguments
from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.common.utils import normalize_bbox, clamp
from core.datasets.collate_self_supervised import DataCollatorForLayoutModeling, DataCollatorForTextAndLayoutReconstruction, DataCollatorForVisualTextRecognition, compute_input_and_target_lengths

logger = logging.get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Split(Enum):
    train = 'train'
    dev = 'dev'
    test = 'test'


class IITCDIPDataset(Dataset):
    args: UdopDataArguments
    mode: Split

    def is_world_process_zero(self):# -> bool:
        return self.args.local_rank in [0, -1] or torch.distributed.get_rank() == 0

    def __init__(
        self,
        args: UdopDataArguments,
        tokenizer: PreTrainedTokenizer,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        task: Optional[str] = None,
    ):
        
        """
            Structure of data directory: 
            
            args.data_dir             
                ├── cdip-images                     # OCR Data and Document Images
                └── cdip_stat.txt.14627461.list     # IIT-CDIP Meta Files
        """       
            
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError('mode is not a valid split name')

        self.args = args
        self.mode = mode
        self.data_dir = os.path.join(args.data_dir, args.mpdfs_dir)
        self.tokenizer = tokenizer
        self.task = task
         
        self.max_token_len = self.args.max_seq_length - self.tokenizer.num_special_tokens_to_add()
        self.rng = random.Random(args.seed)
        self.img_trans = img_trans_torchvision
        self.ocr_folder_name = args.ocr_dir    # 'cdip-images-full-clean-ocr021121'
        self.img_dir = args.img_dir
        self.image_size = args.image_size
        self.use_line_tokens = args.use_line_tokens
        self.index_wrapper = list(range(len(self.examples)))
        self.whole_word_masking = task in ['TextAndLayoutReconstruction', 'LayoutModeling', 'VisualTextRecognition']\
        and args.whole_word_masking
        self.image_drop_rate = args.image_drop_rate
        
        # Load Meta Data
        self.examples = []
        index_file = args.train_file if mode == Split.train else args.dev_file
        ocr_folder_names = self.ocr_folder_name.split("|") if self.ocr_folder_name else None

        for i, one_index_file in enumerate(index_file.split("|")):
            ocr_folder_name = ocr_folder_names[i] if ocr_folder_names else None
            self.examples += self.load_from_index_file(os.path.join(args.data_dir, args.mpdfs_dir, one_index_file), ocr_folder_name)
            
        # Load Self Supervised Tasks Collator
        if task == 'TextAndLayoutReconstruction': 
            self.ocr_collator = DataCollatorForTextAndLayoutReconstruction(
                tokenizer=tokenizer,
                noise_density=args.mlm_probability,
                mean_noise_span_length=args.mean_noise_span_length,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=0,
            )
        elif task == 'LayoutModeling':     
            self.ocr_collator = DataCollatorForLayoutModeling(
                tokenizer=tokenizer,
                noise_density=args.mlm_probability,
                mean_noise_span_length=args.mean_noise_span_length,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=0,
            )
        elif task == 'VisualTextRecognition': 
            
            self.ocr_collator = DataCollatorForVisualTextRecognition(
                tokenizer=tokenizer,
                noise_density=0.75,
                mean_noise_span_length=4,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=0,
            )
        else:
            raise
            
    def load_from_index_file(self, index_file, ocr_folder_name):
        with open(index_file, 'r') as f:
            examples = []
            for line in f:
                line = line.strip().split()

                if len(line) == 4:
                    lang, url, fname, pid = line
                elif len(line) == 1:
                    fname = line[0]
                    if not fname.endswith(".ocr.json"):
                        fname += ".ocr.json"
                    pid = 0
                elif len(line) == 2:
                    fname, pid = line
                else:
                    continue
                if ocr_folder_name:
                    fname = os.path.join(ocr_folder_name, fname)
                examples.append((fname, pid))

            return examples


    def load_oneocr_example(self, fname, return_ocr=True, return_img=True):
        ocr_path = os.path.join(self.data_dir, fname)

        ocr = None
        if return_ocr:
            with open(ocr_path, 'r', encoding='utf8') as f:
                ocr = json.load(f)
        img = None
        if return_img:
            ocr_folder_names = self.ocr_folder_name.split("|") if self.ocr_folder_name else None
            img_path = os.path.join(self.data_dir, fname.replace('.ocr.json', ''))

            if ocr_folder_names:
                for ocr_folder_name in ocr_folder_names:
                    if ocr_folder_name in img_path:
                        img_path = fname.replace(ocr_folder_name, self.img_dir, 1).replace('.ocr.json', '')
                        break

            img_dir, img_name = os.path.split(img_path)
            img_basename = os.path.splitext(img_name)[0]
            img_path = os.path.join(self.data_dir, img_path)

            if not os.path.exists(img_path):
                for img_ext in ['tif', 'tiff', 'png', 'jpg', 'jpeg']:
                    img_path = os.path.join(self.data_dir, img_dir, img_basename + '.' + img_ext)
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        
                        if img.format.lower() in ['jpeg', 'png', 'tiff']:
                            break
            else:
                img = Image.open(img_path)
            if img is not None:
                assert (img.format.lower() in ['jpeg', 'png', 'tiff'])

        return ocr, img
    
    def load_example(self, item, return_ocr=True, return_img=True):
        fname, pid = self.examples[item]

        if fname.endswith('.json'):
            ocr, img = self.load_oneocr_example(fname, return_ocr, return_img)
            return ocr, img, 'json'
        else:
            logger.warning(f'The format of {fname} is not supported.')
            return None
        
    def get_bb(self, bb, page_size, normalize=True):
        bbs = [float(j) for j in bb]
        xs, ys = [], []
        for i, b in enumerate(bbs):
            if i % 2 == 0:
                xs.append(b)
            else:
                ys.append(b)
        (width, height) = page_size
        return_bb = [
            clamp(min(xs), 0, width - 1),
            clamp(min(ys), 0, height - 1),
            clamp(max(xs), 0, width - 1),
            clamp(max(ys), 0, height - 1),
        ]

        if normalize:
            return_bb = normalize_bbox(return_bb, page_size)
        return return_bb

    def random_sample_oneocr(self, doc, img, task='mlm'):
        
        if doc is not None:
            available_list = []
            for i in range(len(doc)):
                if not self.no_img:
                    try:
                        img.seek(i)
                        assert (len(doc[i]['lines']) > 0)
                    except:
                        continue
                available_list.append(i)
            if len(available_list) == 0:
                raise RuntimeError('empty doc-img pair: ' + img.filename if hasattr(img, 'filename') else '')
            pid = available_list[self.rng.randint(0, len(available_list) - 1)]
            page = doc[pid]
            lines = page['lines']
            text_list, bbox_list = [], []
            line_bbox_list = []
            line_bbox_content_list = []
            char_list = []
            char_bbox_list = []
            page_token_cnt = 0
            if lines != []:
                height, width = float(page['height']), float(page['width'])
                page_size = (width, height)
                for cnt, line in enumerate(lines):
                    line_content = []
                    for j, word in enumerate(line['words']):
                        text = self.normal_text(word['text'])
                        if text == '':
                            continue
                        bb = self.get_bb(word['boundingBox'], page_size)
                        bb = [item/1000.0 for item in bb]
                        sub_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        if self.whole_word_masking:
                            text_list.append(sub_tokens)
                            bbox_list.append([bb] * len(sub_tokens))
                        else:
                            text_list.extend(sub_tokens)
                            bbox_list.extend([bb] * len(sub_tokens))
                        line_content.extend(list(range(page_token_cnt, page_token_cnt + len(sub_tokens))))
                        page_token_cnt += len(sub_tokens)
                        chars = self.tokenizer.encode(list(word['text']), add_special_tokens=False)
                        char_list.extend(chars)
                        # print(list(word['text']), word['text'], chars)
                        char_bbox_list.extend([bb] * len(chars))

                    if len(line_content) > 0:
                        line_bb = self.get_bb(line['boundingBox'], page_size)
                        line_bbox_list.append(line_bb)
                        line_bbox_content_list.append(line_content)

            if len(text_list) <= self.max_token_len:
                start = 0
                end = len(text_list)
            else:
                start = self.rng.randint(0, len(text_list) - self.max_token_len - 1)
                end = start + self.max_token_len

            line_bbox_list_ret = []
            line_bbox_content_list_ret = []
            for i in range(len(line_bbox_content_list)):
                cur_list_content = [
                    x - start for x in line_bbox_content_list[i]
                    if start <= x < end
                ]
                if len(cur_list_content) > 0:
                    line_bbox_list_ret.append(line_bbox_list[i])
                    line_bbox_content_list_ret.append(cur_list_content)
            assert (len(line_bbox_list_ret) == len(line_bbox_content_list_ret))
        page_img = None
        
        if img is not None:
            if doc is None:
                try:
                    pid = self.rng.randint(0, img.n_frames-1)
                except:
                    pid = 0
                    
            img.seek(pid)
            page_img = self.img_trans(img, self.image_size)

        if doc is None:
            return page_img
        
        return (text_list[start:end], bbox_list[start:end], line_bbox_list_ret, line_bbox_content_list_ret, page_img, img, char_list, char_bbox_list)

        
    def get_item(self, item):

        ocr, img, format = self.load_example(item)

        char_list = [0]
        char_bbox_list = [[0,0,0,0]]
        if format == 'json':
            doc = ocr['analyzeResult']['readResults']
            text_list, bbox_list, line_bbox_list, line_bbox_content_list, page_img, img, char_list, char_bbox_list = self.random_sample_oneocr(doc, img, task=self.task)
        else:
            raise ValueError('None format')
                
        max_length = self.args.max_seq_length
        
        image_mask_label = None
        ids_restore = None
        ids_keep = None
        
        '''
        # Loading Line ids
        for i in range(len(line_bbox_content_list)):
            line_bbox_content_list[i] = [x + 1 for x in line_bbox_content_list[i]]
        line_ids = torch.zeros_like(model_input['input_ids'])
        for idx_line, line in enumerate(line_bbox_content_list, start=1):
            for idx_token in line:
                line_ids[idx_token] = idx_line
        line_ids = torch.cat([
            line_ids,
            torch.zeros(self.args.max_seq_length - line_ids.size(0),
                        dtype=line_ids.dtype)
        ])
        '''

        visual_bbox_input = get_visual_bbox(self.image_size)
        
        if self.task in ['text_and_layout_reconstruction', 'layout_modeling', 'visual_text_recognition']:
            if len(text_list) == 1:              
                text_list = text_list[0]
                bbox_list = bbox_list[0]
            if random.random() < self.image_drop_rate:
                page_img = torch.ones_like(page_img) * torch.mean(page_img)
            input_ids, labels, bbox_list = self.ocr_collator(text_list, bbox_list, self.task)
        else:
            raise
            
        attention_mask = [1] * len(input_ids)
        decoder_attention_mask = [1] * len(labels)

        bbox_input = torch.tensor(bbox_list, dtype=torch.float)
        visual_bbox_input = torch.tensor(visual_bbox_input, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
        char_ids = torch.tensor(char_list, dtype=torch.long)
        char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)
        
        assert(len(bbox_input) == len(input_ids))
        assert(len(labels) > 0)
        assert input_ids is not None
        assert page_img is not None
        
        inputs = {
            'input_ids': input_ids,
            'visual_seg_data': visual_bbox_input,
            #'line_ids': line_ids,
            'seg_data': bbox_input,
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'image': page_img,
            'image_mask_label': image_mask_label,
            'ids_restore': ids_restore,
            'ids_keep': ids_keep,
            'char_ids': char_ids,
            'char_seg_data': char_bbox_input
        }

        return inputs, locals()

    def normal_text(self, t):
        if type(t) is float:
            if t == int(t):
                t = int(t)
        t = str(t)
        return t.strip()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        while True:
            try:
                inputs, locals_dict = self.get_item(item)
                break
            except Exception as e:
                logger.warning(e)
                traceback.print_exc()
                new_item = self.rng.randint(0, self.__len__() - 2)
                if new_item >= item:
                    new_item += 1
                item = new_item

        return inputs
