import math
import collections
import pickle
import os
from random import shuffle

import numpy as np
from transformers import PreTrainedTokenizerBase


class DataCollatorForT5DocCLS:
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, ori_input_ids, ori_bbox_list, labels):

        prompt_text = 'document classification.'
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

        labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list


class DataCollatorForT5DocIE:
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        self.tokenizer = tokenizer
        
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples, prompt_text='named entity recognition'):     
        
        prompt_text = f'{prompt_text}.'
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tokens = examples['context']
        input_ids = []
        bboxes= []
        for i in range(len(input_tokens)):
            subtokens = self.tokenizer.encode(input_tokens[i], add_special_tokens=False)
            input_ids += subtokens
            bboxes += examples['bboxes'][i:i+1] * len(subtokens)
        input_ids = prompt_ids + input_ids + [1]
        bboxes = [[0,0,0,0]] * len(prompt_ids) + bboxes + [[1.0, 1.0, 1.0, 1.0]]
        
        label_tokens = ''
        for key in examples['key_value']:
            entity = examples['key_value'][key]
            label_tokens += f'{key} {entity}'      
            
        labels = self.tokenizer.encode(label_tokens, add_special_tokens=True)
        
        return input_ids, labels, bboxes


class DataCollatorForT5DocLayout:
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        self.tokenizer = tokenizer
        
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples, prompt_text='layout analysis'):             

        prompt_text = f'{prompt_text}. '
        labels_text = ''
        for i, entity in enumerate(examples['key_value']):
            prompt_text += f'{entity} '
            labels_text += f'{entity} '
            for bbox in examples['key_value'][entity]:
                x0, y0, x1, y1 = (np.array(bbox) * (self.tokenizer._loc_extra_ids - 1)).astype(int)
                labels_text += f'<loc_{x0}><loc_{y0}><loc_{x1}><loc_{y1}>'
        prompt_text = prompt_text[:-1] + '.'        
        labels = self.tokenizer.encode(labels_text.strip(), add_special_tokens=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        if 'context' in examples:
            input_tokens = examples['context']
            input_ids = []
            bboxes= []
            for i in range(len(input_tokens)):
                subtokens = self.tokenizer.encode(input_tokens[i], add_special_tokens=False)
                input_ids += subtokens
                bboxes += examples['bboxes'][i:i+1] * len(subtokens)
            input_ids = prompt_ids + input_ids + [1]
            bboxes = [[0,0,0,0]] * len(prompt_ids) + bboxes + [[1.0, 1.0, 1.0, 1.0]]
        else:
            input_ids = prompt_ids + [1]
            bboxes = [[0,0,0,0]] * len(prompt_ids) + [[1.0, 1.0, 1.0, 1.0]]
        return input_ids, labels, bboxes


class DataCollatorForT5DocQA:
    """
    Data collator used for T5 document information extraction
    """
    def __init__(self, tokenizer=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        self.tokenizer = tokenizer
        
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, example, prompt_text='question answering'):             
                      
        question = example['question']
        answer = example['answer']
        
        prompt_text = f'{prompt_text}. {question}'        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        labels_text = f'{answer}'
        labels = self.tokenizer.encode(labels_text, add_special_tokens=True)
        
        if 'context' in example.keys():
            bbox_sub = []
            inputs_sub = []
            labels_sub = []
        
            input_text = example['context']
            bboxes = example['bboxes']
            for i in range(len(input_text)):
                subtokens = self.tokenizer.encode(input_text[i], add_special_tokens=False)
                inputs_sub += subtokens
                bbox_sub += bboxes[i:i+1] * len(subtokens)
                
            input_ids = prompt_ids + inputs_sub + [1]
            bbox = [[0,0,0,0]] * (len(prompt_ids)) + bbox_sub + [[1.0,1.0,1.0,1.0]]
        
        else:
            input_ids = prompt_ids + [1]
            bbox = [[0,0,0,0]] * len(prompt_ids) + [[1.0,1.0,1.0,1.0]]
        return input_ids, labels, bbox

    
class DataCollatorForT5DocDue:
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):
        self.tokenizer = tokenizer
        
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, examples, prompt_text='layout analysis'):             

        prompt_text = f'{prompt_text}.'
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tokens = examples['context']
        input_ids = []
        bboxes = []
        
        label_tokens = ''
        key_ids = []
        for key in examples['key_value']:
            entity = examples['key_value'][key][0]
            key_ids += self.tokenizer.encode(key, add_special_tokens=False)
            label_tokens += f'{entity} '      
        labels = self.tokenizer.encode(label_tokens[:-1], add_special_tokens=True)
        
        for i in range(len(input_tokens)):
            subtokens = self.tokenizer.encode(input_tokens[i], add_special_tokens=False)
            input_ids += subtokens
            bboxes += examples['bboxes'][i:i+1] * len(subtokens)
            
        input_ids = prompt_ids + key_ids + input_ids + [1]
        bboxes = [[0,0,0,0]] * len(prompt_ids) + [[0,0,0,0]] * len(key_ids) + bboxes + [[1.0, 1.0, 1.0, 1.0]]
        return input_ids, labels, bboxes    