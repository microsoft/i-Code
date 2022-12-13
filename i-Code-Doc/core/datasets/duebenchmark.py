import json
import logging
import os
import collections
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from core.common.utils import img_trans_torchvision, get_visual_bbox
from core.datasets.collate_supervised import DataCollatorForT5DocDue

logger = logging.getLogger(__name__)


def get_value(annotation_value):
    if 'value_variants' in annotation_value:
        return annotation_value['value_variants']
    else:
        return [annotation_value['value']]

def get_child_values(annotation_values):
    values: List = []
    for annotation_value in annotation_values:
        values += [annotation_value['value']]

    return values

def _get_page_dim(pages_dim):
    # function will return True if all pages are with the same dimension
    # and dimension of mostly use dimension in the document
    tpages = [tuple(a) for a in pages_dim]
    counter = collections.Counter(tpages)

    return len(counter) == 1, counter.most_common(1)[0][0]
    
def _convert_to_relative_page_pos(seg, seg_key, page_seg, page_same_size, page_most_dim):
    # positions - n x 4 array
    # page_positions = 1 x 4 array
    rel_pos = np.zeros_like(seg, dtype=float)
    rel_pos[:, 0] = seg[:, 0].clip(min=0, max=page_most_dim[2]) / (page_most_dim[2] * np.sqrt(2))
    rel_pos[:, 2] = seg[:, 2].clip(min=0, max=page_most_dim[2]) / (page_most_dim[2] * np.sqrt(2))
    rel_pos[:, 1] = seg[:, 1].clip(min=0, max=page_most_dim[3]) / (page_most_dim[2] * np.sqrt(2))
    rel_pos[:, 3] = seg[:, 3].clip(min=0, max=page_most_dim[3]) / (page_most_dim[2] * np.sqrt(2))

    if not page_same_size:
        for psize, prange in zip(page_seg['org_bboxes'], page_seg['ranges']):
            if tuple(psize) == page_most_dim:
                continue
            # get index of bboxes which need to be scaled with different dimension
            if seg_key == 'tokens':
                idx = list(range(prange[0], prange[1]))
            else:
                idx = []
                for seg_idx, seg_rng in enumerate(seg['ranges']):
                    if prange[0] <= seg_rng[0] < prange[1]:
                        idx.append(seg_idx)

            rel_pos[idx, 0] = seg[idx, 0].clip(min=0, max=psize[2]) / (psize[2] * np.sqrt(2))
            rel_pos[idx, 2] = seg[idx, 2].clip(min=0, max=psize[2]) / (psize[2] * np.sqrt(2))
            rel_pos[idx, 1] = seg[idx, 1].clip(min=0, max=psize[3]) / (psize[2] * np.sqrt(2))
            rel_pos[idx, 3] = seg[idx, 3].clip(min=0, max=psize[3]) / (psize[2] * np.sqrt(2))

    return rel_pos


class Duebenchmark(Dataset):

    def __init__(self, data_args, tokenizer, mode='train'):
        
        """
            Structure of data directory: 
            
            args.data_dir             
                ├── DocVQA            # DocVQA Images
                ├── InfographicsVQA   # InfographicsVQA Images
                ├── aws_neurips_time  
                ├──        ├── docvqa            # DocVQA Meta Files
                ├──        ├── infographics_vqa  # InfographicsVQA Meta Files
                └── ...
        """       
         
        assert os.path.isdir(data_args.data_dir), f"Data dir {data_args.data_dir} does not exist!"
        self.data_dir = data_args.data_dir
        self.duebenchmark_dir = data_args.duebenchmark_dir
        self.max_seq_length = data_args.max_seq_length
        self.max_seq_length_decoder = data_args.max_seq_length_decoder
        self.image_size = data_args.image_size

        # Load Meta Data
        meta_path = os.path.join(data_args.data_dir, data_args.duebenchmark_dir, 'examples.npy')
        if not os.path.exists(meta_path):
            task_meta_paths = ['docvqa', 'infographics_vqa', 'KleisterCharity', 'AxCell', 'DeepForm', 'TabFact', 'WikiTableQuestions']
            task_pdf_paths = ['DocVQA', 'InfographicsVQA', 'KleisterCharity', 'PWC', 'DeepForm', 'TabFact', 'WikiTableQuestions']
            task_names = ['question answering on DocVQA', 'question answering on InfographicsVQA', 'information extraction on KleisterCharity', 'information extraction on PWC', 'information extraction on DeepForm', 'table natural language inference on TabFact', 'table question answering on WikiTableQuestions']
            task_meta_paths = task_meta_paths[2:3]
            task_pdf_paths = task_pdf_paths[2:3]
            task_names = task_names[2:3]
            self.examples = []
            for i in range(len(task_meta_paths)):
                task_meta_path = os.path.join(data_args.data_dir, data_args.duebenchmark_dir, 'aws_neurips_time', task_meta_paths[i])
                task_meta_path_context = os.path.join(task_meta_path, mode, 'documents_content.jsonl')
                task_meta_path_label = os.path.join(task_meta_path, mode, 'document.jsonl')
                task_pdf_path = os.path.join('.', task_pdf_paths[i])
                meta = [json.loads(jline) for jline in tqdm(open(task_meta_path_context).readlines())]
                label_meta = [json.loads(jline) for jline in tqdm(open(task_meta_path_label).readlines())]

                for j in range(len(meta)):
                    key = meta[j]['name']
                    image_path = os.path.join(task_pdf_path, key + '.pdf')
                    context = meta[j]['contents'][0]['tokens_layer']['tokens']
                    bboxes = meta[j]['contents'][0]['tokens_layer']['positions']
                    
                    if bboxes != []:
                        bboxes = np.array(bboxes)
                        page_seg = {}
                        page_seg['ranges'] = np.array(meta[j]['contents'][0]['tokens_layer']['structures']['pages']['structure_value'])
                        page_seg['org_bboxes'] = np.array(meta[j]['contents'][0]['tokens_layer']['structures']['pages']['positions'])
                        bboxes = np.array(bboxes)
                        page_same_size, page_most_dim = _get_page_dim(bboxes)
                        bboxes = _convert_to_relative_page_pos(bboxes, 'tokens', page_seg, page_same_size, page_most_dim)
                        bboxes = bboxes.tolist()
        
                    for annotation in label_meta[i]['annotations']:
                        example = {}
                        example['image_path'] = image_path
                        example['context'] = context
                        example['bboxes'] = bboxes
                        example['task_name'] = task_names[i]
                        example_kv = defaultdict(list)
                        key = annotation['key']
                        values = []
                        for _, value in enumerate(annotation['values']):
                            if 'children' in value:
                                for child in value['children']:
                                    child_question = f"What are the {key} values for the {child['key']} column?"
                                    example_kv[child_question] += get_child_values(child['values'])
                            else:
                                values += get_value(value)

                        if values is not None:
                            example_kv[key] = values
                        example['key_value'] = example_kv
                        self.examples.append(example)
            np.save(meta_path, np.array(self.examples))
        else:
            self.examples = np.load(meta_data, allow_pickle=True)
            
        # Load Due Benchmark Format Collator
        self.due_collator = DataCollatorForT5DocDue(
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

            image = convert_from_path(os.path.join(self.data_dir, self.duebenchmark_dir, example['image_path']))[0]
            image_ = img_trans_torchvision(image, self.image_size)
            visual_bbox_input = get_visual_bbox(self.image_size)
            
            image_mask_label = None
            ids_restore = None
            ids_keep = None
            char_list = [0]
            char_bbox_list = [[0,0,0,0]]
            if self.task == 'due':
                input_ids, labels, bbox_list = self.due_collator(example, prompt_text=example['task_name'])
            
            assert len(text_list) > 0
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

