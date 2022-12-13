import logging
import os
from itertools import groupby

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocIE, DataCollatorForT5DocLayout

logger = logging.getLogger(__name__)


def layout_collate_bbox(bboxes):
    x0, y0, x1, y1 = 1.0, 1.0, 0.0, 0.0
    for bbox in bboxes:
        x0, y0, x1, y1 = min(bbox[0], x0), min(bbox[1], y0), max(bbox[2], x1), max(bbox[3], y1)
    return [x0, y0, x1, y1]


class Docbank(Dataset):

    def __init__(self, data_args, tokenizer, 
                 mode='train', task='layout'):
        
        """
            Structure of data directory: 
            
            args.data_dir             
                ├── DocBank_500K_ori_img       # DocBank Images
                └── DocBank_500K_txt           # DocBank Meta Files
        """       
         
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        self.task = task
        self.data_dir = data_args.data_dir
        self.docbank_dir = data_args.docbank_dir
        self.image_size = data_args.image_size

        # Load Meta Data
        meta_path = os.path.join(data_args.data_dir, data_args.docbank_dir, 'examples.npy')
        if not os.path.exists(meta_path) or 1:
            example_paths = os.listdir(os.path.join(data_args.data_dir, data_args.docbank_dir, 'DocBank_500K_txt'))
            self.examples = []
            for path in tqdm(example_paths):
                self.examples += [{'data_path': os.path.join('DocBank_500K_txt', path), 'image_path': os.path.join('DocBank_500K_ori_img', path.replace('.txt', '_ori.jpg'))}]
            np.save(meta_path, np.array(self.examples))
        else:
            self.examples = np.load(meta_path, allow_pickle=True)
            
        # Load Layout Analysis and Information Extraction Collator
        if task == 'layout':
            self.layout_collator = DataCollatorForT5DocLayout(
                    tokenizer=tokenizer,
                    input_length=data_args.max_seq_length,
                    target_length=data_args.max_seq_length_decoder,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=0,
                )
        elif task == 'ie':
            self.ie_collator = DataCollatorForT5DocIE(
                    tokenizer=tokenizer,
                    input_length=data_args.max_seq_length,
                    target_length=data_args.max_seq_length_decoder,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=0,
                )
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        try:
            example = self.examples[index]

            data = pd.read_table(os.path.join(self.data_dir, self.docbank_dir, example['data_path']), header=None, names=['token','x0','y0','x1','y1','R','G','B','name', 'label'])
            data = data.replace({np.nan: '<unk>'})

            image = Image.open(os.path.join(self.data_dir, self.docbank_dir, example['image_path']))
            image_ = img_trans_torchvision(image, self.image_size)
            
            visual_bbox_input = get_visual_bbox(self.image_size)

            image_mask_label = None
            ids_restore = None
            ids_keep = None
        
            char_list = [0]
            char_bbox_list = [[0,0,0,0]]
            if self.task == 'layout':
                ziped = zip(data['label'].to_numpy(), data[['x0', 'y0', 'x1', 'y1']].to_numpy()/1000.0, data['token'].to_numpy())
                grouped_data = [np.array(list(j)) for i, j in groupby(ziped, lambda x: x[0])]
                input_dict = {}
                input_tokens = []
                bboxes = []
                input_dict['key_value'] = {}
                
                for item in grouped_data:
                    entity = item[0,0]
                    bbox = layout_collate_bbox(item[:, 1])
                    if entity not in input_dict:
                        input_dict[entity] = []
                    if entity not in input_dict['key_value']:
                        input_dict['key_value'][entity] = []
                    input_dict['key_value'][entity].append(bbox)
                    
                    bboxes += item[:, 1].tolist()
                    input_tokens += item[:, 2].tolist()
                
                input_dict['context'] = input_tokens
                input_dict['bboxes'] = bboxes
                input_ids, labels, bbox_list = self.layout_collator(input_dict, prompt_text='layout analysis on docbank')
                
            elif self.task == 'ie':
                ziped = zip(data['label'].to_numpy(), data[['x0', 'y0', 'x1', 'y1']].to_numpy()/1000.0, data['token'].to_numpy())
                grouped_data = [np.array(list(j)) for i, j in groupby(ziped, lambda x: x[0])]
                input_dict = {}
                input_tokens = []
                bboxes = []
                input_dict['key_value'] = {}
                
                for item in grouped_data:
                    entity = item[0, 0]
                    bboxes += item[:, 1].tolist()
                    
                    tokens = ' '.join(item[:, 2])
                    input_tokens += [tokens]
                    input_dict['key_value'][tokens] = entity
                    
                input_dict['context'] = input_tokens
                input_dict['bboxes'] = bboxes
                input_ids, labels, bbox_list = self.ie_collator(input_dict, prompt_text='named entity recognition on docbank')
            

            attention_mask = [1] * len(input_ids)
            decoder_attention_mask = [1] * len(labels)

            char_ids = torch.tensor(char_list, dtype=torch.long)
            char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)
            
            bbox_input = torch.tensor(bbox_list, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)

            assert input_ids is not None
            assert len(input_ids) == len(bbox_input)
            assert len(char_bbox_input.size()) == 2
            
            inputs = {
                'input_ids': input_ids,
                'seg_data': bbox_input,
                'visual_seg_data': visual_bbox_input,
                'attention_mask': attention_mask,
                'decoder_attention_mask': decoder_attention_mask,
                'labels': labels,
                'image': image_,
                'char_ids': char_ids,
                'char_seg_data': char_bbox_input,
                'image_mask_label': image_mask_label,
                'ids_restore': ids_restore,
                'ids_keep': ids_keep
            }

            return inputs
        except:
            return self[(index + 1) % len(self)]        

