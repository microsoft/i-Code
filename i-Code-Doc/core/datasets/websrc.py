import json
import logging
import os
import glob

import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from core.common.utils import img_trans_torchvision, get_visual_bbox 
from core.datasets.collate_supervised import DataCollatorForT5DocQA


logger = logging.getLogger(__name__)


class Websrc(Dataset):
    
    def __init__(self, data_args, tokenizer, mode='train', task='qa'):

        """
            Structure of data directory: 
            
            args.data_dir             
                ├── auto        # Meta/Image Files by Category
                ├── book
                ├── ...
                └── dataset_split.csv      # Dataset Split
        """          
        
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        logger.info(f'Loading Websrc')
        self.task = task
        self.data_dir = data_args.data_dir
        self.websrc_dir = data_args.websrc_dir
        self.image_size = data_args.image_size
        
        # Load Metadata
        meta_path = os.path.join(data_args.data_dir, data_args.websrc_dir, 'examples.npy')
        if not os.path.exists(meta_path) or 1:
            self.examples = []
            all_path = list(glob.iglob(os.path.join(data_args.data_dir, data_args.websrc_dir, '*/*')))
            for path in tqdm(all_path):
                tmp = '/'.join(path.split('/')[-2:])
                csv = pd.read_csv(os.path.join(path, 'dataset.csv'))
                for i in range(len(csv)):
                    processed_item = {'question': csv.iloc[i]['question'], 'answer': csv.iloc[i]['answer'], 'image_path': os.path.join(tmp, 'processed_data', csv.iloc[i]['id'][2:9]+'.png')}
                    self.examples.append(processed_item)          
            np.save(meta_path, np.array(self.examples))
        else:
            self.examples = np.load(meta_path, allow_pickle=True)
        
        # Get QA Task Collator
        self.qa_collator = DataCollatorForT5DocQA(
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

            image = Image.open(os.path.join(self.data_dir, self.websrc_dir, example['image_path'])).convert('RGB')

            image = img_trans_torchvision(image, self.image_size)
            visual_bbox_input = get_visual_bbox(self.image_size)

            input_ids, labels, bbox_list = self.qa_collator(example, prompt_text='question answering on websrc')

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
            return inputs
        except:
            return self[(index + 1) % len(self)]

