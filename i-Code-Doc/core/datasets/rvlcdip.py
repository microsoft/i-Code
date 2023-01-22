import json
import logging
import os
import random

from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocCLS

EMPTY_BOX = [0, 0, 0, 0]
SEP_BOX = [1000, 1000, 1000, 1000]

logger = logging.getLogger(__name__)


def normalText(t):
    if type(t) is float:
        if t == int(t):
            t = int(t)
    t = str(t)
    return t.strip()


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


def get_bb(bb):
    bbs = [float(j) for j in bb]
    xs, ys = [], []
    for i, b in enumerate(bbs):
        if i % 2 == 0:
            xs.append(b)
        else:
            ys.append(b)
    return [min(xs), min(ys), max(xs), max(ys)]


def get_rvlcdip_labels():
    return [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budget",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo"
    ]


class RvlCdipDataset(Dataset):
    
    NUM_LABELS = 16

    def __init__(self, data_args, tokenizer, mode='train'):
        
        """ Structure of data directory: 
            
            args.data_dir             
                ├── images
                ├── labels
                    ├── test.txt
                    ├── train.txt
                    └── val.txt
        """                 
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        logger.info('Loading RVL-CDIP')
        
        ocr_dir = os.path.join(data_args.data_dir, data_args.mpdfs_dir, 'cdip-images-full-clean-ocr021121')
        image_dir = os.path.join(data_args.data_dir, data_args.mpdfs_dir, 'cdip-images')
        label_dir = os.path.join(data_args.data_dir, data_args.rvlcdip_dir, 'labels')
        if mode == 'train': 
            filename = 'train.txt'
        elif mode == 'val': 
            filename = 'val.txt'
        elif mode == 'test': 
            filename = 'test.txt'
        else: 
            raise NotImplementedError
        file_path = os.path.join(label_dir, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            file_list = f.read().splitlines()
        
        max_samples = None
        if (max_samples is not None) and (len(file_list) > max_samples): 
            file_list = file_list[:max_samples]


        self.cls_bbox = EMPTY_BOX[:]
        self.pad_bbox = EMPTY_BOX[:]
        self.sep_bbox = SEP_BOX[:]

        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.num_img_embeds = 0
        
        label_list = get_rvlcdip_labels()
        self.label_list = label_list
        self.label_map = dict(zip(list(range(len(self.label_list))), self.label_list))
        self.n_classes = len(label_list)
        self.label_list = label_list

        self.image_size = data_args.image_size
        
        self.examples = []
        self.labels = []
        self.images = []

        self.cls_collator = DataCollatorForT5DocCLS(
                tokenizer=tokenizer,
            )
        
        results = [self.load_file(filepath, ocr_dir, image_dir) for filepath in tqdm(file_list)]
        for labels, examples, images in results: 
            self.labels += labels 
            self.examples += examples 
            self.images += images
        assert len(self.labels) == len(self.examples) 
        logger.info(f'There are {len(self.labels)} images with annotations')

    def load_file(self, file_path, ocr_dir, image_dir): 
        labels, examples, images = [], [], []
        img_path, label = file_path.split()
        
        # label = int(label)
        if img_path.endswith('.tif'):
            file_path = img_path + '.ocr.json'
            file_path = os.path.join(ocr_dir, file_path)
            image = os.path.join(image_dir, img_path)
#             if os.path.exists(file_path):
            labels.append(label)
            examples.append(file_path)
            images.append(image)
        else:
            raise NotImplementedError

        return labels, examples, images

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        try:
            label = self.labels[index]
            label = self.label_map[int(label)]

            rets, n_split = read_ocr_core_engine(self.examples[index], self.images[index], self.tokenizer, self.max_seq_length, self.num_img_embeds, self.image_size)
            if n_split == 0:
                # Something wrong with the .ocr.json file
                print("EMPTY ENTRY")
                return self[(index + 1) % len(self)]
            for i in range(n_split):
                text_list, bbox_list, image, page_size = rets[i]
                (width, height) = page_size
                bbox = [
                    [
                        b[0] / width,
                        b[1] / height,
                        b[2] / width,
                        b[3] / height,
                    ]
                    for b in bbox_list
                ]
                
                visual_bbox_input = get_visual_bbox(self.image_size)

                input_ids = self.tokenizer.convert_tokens_to_ids(text_list)

                input_ids, labels, bbox_input = self.cls_collator(input_ids, bbox, label)
                attention_mask = [1] * len(input_ids)
                decoder_attention_mask = [1] * len(labels)

                char_list = [0]
                char_bbox_list = [[0,0,0,0]]
                char_ids = torch.tensor(char_list, dtype=torch.long)
                char_bbox_input = torch.tensor(char_bbox_list, dtype=torch.float)
               
                bbox_input = torch.tensor(bbox_input, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
                assert len(bbox_input) == len(input_ids)
                assert len(bbox_input.size()) == 2
                assert len(char_bbox_input.size()) == 2
                
                return_dict =  {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "seg_data": bbox_input,
                    "visual_seg_data": visual_bbox_input,
                    "decoder_attention_mask": decoder_attention_mask,
                    "image": image,
                    'char_ids': char_ids,
                    'char_seg_data': char_bbox_input
                }
                assert input_ids is not None

                return return_dict
        except:
            return self[(index + 1) % len(self)]

    def get_labels(self):
        return list(map(str, list(range(self.NUM_LABELS))))

    def pad_tokens(self, input_ids, bbox):
        # [CLS], sentence, [SEP]
        tokenized_tokens = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        start_token, _, end_token = tokenized_tokens[0], tokenized_tokens[1:-1], tokenized_tokens[-1]
        
        sentence = tokenized_tokens 
        expected_seq_length = self.max_seq_length - self.num_img_embeds
        mask = torch.zeros(expected_seq_length)
        mask[:len(sentence)] = 1

        bbox = [self.cls_bbox] + bbox + [self.sep_bbox]
        while len(sentence) < expected_seq_length:
            sentence.append(self.tokenizer.pad_token_id)
            bbox.append(self.pad_bbox)
        
        assert len(sentence) == len(bbox)
        return (sentence, mask, bbox, start_token, end_token) 

    
# Might need to change to your own OCR processing
def read_ocr_core_engine(file, image_dir, tokenizer, max_seq_length, num_img_embeds, image_size):
    with open(file, 'r', encoding='utf8') as f:
        try:
            data = json.load(f)
        except:
            data = {}
    rets = []
    n_split = 0

    if 'analyzeResult' not in data or 'readResults' not in data['analyzeResult']:
        return rets, n_split

    tiff_images = Image.open(image_dir)

    doc = data['analyzeResult']['readResults']
    
    pid = random.choice(list(range(len(doc))))
    page = doc[pid]
    text_list, bbox_list = [], []
    lines = page['lines']
    height, width = float(page['height']), float(page['width'])
    page_size = (width, height)

    tiff_images.seek(pid)
    image = img_trans_torchvision(tiff_images, image_size)
    for cnt, line in enumerate(lines):
        for j, word in enumerate(line["words"]):
            text = normalText(word['text'])
            if text == '':
                continue
            bb = get_bb(word['boundingBox'])
            sub_tokens = tokenizer.tokenize(text)
            for sub_token in sub_tokens:
                text_list.append(sub_token)
                bbox_list.append(bb)
    if len(text_list) > 0:
        rets.append([text_list, bbox_list, image, page_size])

    assert len(text_list) == len(bbox_list)
    n_split = len(rets)
    
    return rets, n_split
