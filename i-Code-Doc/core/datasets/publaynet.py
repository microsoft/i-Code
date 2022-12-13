import json
import logging
import os

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocLayout

logger = logging.getLogger(__name__)


class Publaynet(Dataset):

    def __init__(self, data_args, tokenizer,
                 mode='train', task='layout'):
        
        """
            Structure of data directory: 
            
            args.data_dir             
                ├── train        # Train Images
                ├── val          # Val Images
                ├── train.json   # Original Train Meta Files
                └── val.json     # Original Val Meta Files
        """        
        
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        logger.info(f'Loading Publaynet')
        label_dir = os.path.join(data_args.data_dir, data_args.publaynet_dir)
        if mode == 'train': 
            filename = 'train.json'
        elif mode == 'val': 
            filename = 'val.json'
        else: 
            raise NotImplementedError
        file_path = os.path.join(label_dir, filename)
        image_dir = os.path.join(label_dir, mode)
        self.image_dir = image_dir
        self.image_size = data_args.image_size
        self.max_seq_length = data_args.max_seq_length

        # Load Meta Data
        meta_data = os.path.join(data_args.data_dir, data_args.publaynet_dir, 'examples.npy')
        if not os.path.exists(meta_data):
            with open(file_path, "r") as f:
                self.data = json.load(open(file_path))


            labels_list = self.data['categories']
            labels_map = {}
            for label in labels_list:
                labels_map[label['id']]=label['name'] 

            self.examples = {}
            self.image_list = self.data['images']
            for image in self.image_list:
                if not image['id'] in self.examples:
                    self.examples[image['id']] = {}              
                    self.examples[image['id']]['layout'] = {}
                self.examples[image['id']]['image_info']=(image['file_name'], [image['width'], image['height']])
                
            for item in tqdm(self.data['annotations']):
                if not item['image_id'] in self.examples:
                    self.examples[item['image_id']] = {}                
                    self.examples[item['image_id']]['layout'] = {}
                    
                _, (width, height) = self.examples[item['image_id']]['image_info']
                
                key = labels_map[item['category_id']]
                if not key in self.examples[item['image_id']]['layout'].keys():
                    self.examples[item['image_id']]['layout'][key] = []
                    
                x0, y0, w, h = item['bbox']
                x0, y0, x1, y1 = x0, y0, x0+w, y0+h
                x0, y0, x1, y1 = x0/width, y0/height, x1/width, y1/height
                self.examples[item['image_id']]['layout'][key].append([x0, y0, x1, y1])

            self.image_map = {}
            self.examples = list(self.examples.values())
            np.save(os.path.join(data_args.data_dir, data_args.publaynet_dir, 'examples.npy'), np.array(self.examples))
        else:
            self.examples = np.load(meta_data, allow_pickle=True)

        # Load Layout Analysis Task Collator
        self.layout_collator = DataCollatorForT5DocLayout(
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

            img_path, (width, height) = example['image_info']
                
            image = Image.open(os.path.join(self.image_dir, img_path))
            image = img_trans_torchvision(image, self.image_size)
            
            visual_bbox_input = get_visual_bbox(self.image_size)

            input_dict = {'key_value': example['layout']}
            input_ids, labels, bbox_list = self.layout_collator(input_dict, prompt_text='layout analysis on publaynet')

            attention_mask = [1] * len(input_ids)
            decoder_attention_mask = [1] * len(labels)

            char_list = [0]
            char_bbox_list = [[0,0,0,0]]
            char_ids = torch.tensor(char_list, dtype=torch.long)
            char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)

            bbox_input = torch.tensor(bbox_list, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
            
            inputs = {
                'input_ids': input_ids,
                'seg_data': bbox_input,
                'visual_seg_data': visual_bbox_input,
                'attention_mask': attention_mask,
                'decoder_attention_mask': decoder_attention_mask,
                'labels': labels,
                'image': image,
                'char_ids': char_ids,
                'char_seg_data': char_bbox_input
            }
            assert input_ids is not None
            assert len(input_ids) == len(bbox_input)
            
            return inputs
        except:
            return self[(index + 1) % len(self)]


