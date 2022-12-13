import json
import logging
import os

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
    
from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocQA

logger = logging.getLogger(__name__)


def process_bbox(bbox):
    x, y, w, h = list(bbox.values())
    return [x, y, x+w, y+h]

def preprocess_sample(sample):
    all_samples = []
    ocr_text = []
    bboxes = []
    for item in sample['bounding_boxes']:
        if item['structure'] not in ['Image', 'Data']:
            ocr_text += [item_['word'] for item_ in item['ocr_info']]
            bboxes += [process_bbox(item_['bbox']) for item_ in item['ocr_info']]
    
    for item in sample['qa_data']:
        sample_ = {}
        sample_['context'] = ocr_text
        sample_['bboxes'] = bboxes
        sample_['question'] = item['question']['text']
        sample_['answer'] = item['answer']['text']
        sample_['image_filename'] = sample['image_filename']
        all_samples.append(sample_)

    return all_samples


class VisualMRC(Dataset):
    

    def __init__(self, data_args, tokenizer, mode='train', task='qa'):
        
        """
            Structure of data directory: 
            
            args.data_dir             
                ├── data        # Original Meta Files
                │    ├── train.jsonl
                │    ├── val.jsonl
                │    └── test.jsonl
                ├── screenshot  # Original Images 
                └── images      # Images (Used)
        """              
        
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        logger.info(f'Loading VisualMRC')
        label_dir = os.path.join(data_args.data_dir, data_args.visualmrc_dir)
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.image_size = data_args.image_size
        self.task = task
        
        if mode == 'train': 
            filename = 'data/train.jsonl'
        elif mode == 'val': 
            filename = 'data/val.jsonl'
        else: 
            filename = 'data/test.jsonl'
            
        file_path = os.path.join(label_dir, filename)
        image_dir = os.path.join(label_dir, mode)

        # Load Metadata
        meta_path = os.path.join(data_args.data_dir, data_args.visualmrc_dir, 'examples.npy')
        if not os.path.exists(meta_path) or 1:
            meta = [json.loads(jline) for jline in tqdm(open(file_path).readlines())]
            self.examples = []
            for item in tqdm(meta):
                self.examples += preprocess_sample(item)
            np.save(meta_path, np.array(self.examples))
        else:
            self.examples = np.load(meta_path, allow_pickle=True)
            
        # Load QA Task Collator
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

            img_path = example['image_filename']
            image = Image.open(os.path.join(self.label_dir, img_path))
            image = img_trans_torchvision(image, self.image_size)
            
            visual_bbox_input = get_visual_bbox(self.image_size)

            image_mask_label = None
            ids_restore = None
            ids_keep = None
            char_list = [0]
            char_bbox_list = [[0,0,0,0]]
            
            input_ids, labels, bbox_list = self.qa_collator(example, prompt_text='question answering on visualmrc' )

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
            assert len(char_bbox_list) > 1

            inputs = {
                'input_ids': input_ids,
                'seg_data': bbox_input,
                'visual_seg_data': visual_bbox_input,
                'attention_mask': attention_mask,
                'decoder_attention_mask': decoder_attention_mask,
                'labels': labels,
                'image': image,
                'char_ids': char_ids,
                'char_seg_data': char_bbox_input,
                'image_mask_label': image_mask_label,
                'ids_restore': ids_restore,
                'ids_keep': ids_keep
            }
            assert input_ids is not None
            assert len(input_ids) == len(bbox_input)
            return inputs
        except:
            return self[(index + 1) % len(self)]

