import math
import collections
import os
from random import shuffle
import random
import numpy as np


def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


class DataCollatorForSelfSupervised:
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, noise_density=0.15, mean_noise_span_length=3, pad_token_id=0, decoder_start_token_id=0, image_size=1024):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_loc_id = self.tokenizer._loc_extra_ids - 1
        
        self.image_size = image_size
        self.patch_size = 16
        self.num_patches = self.image_size//self.patch_size
        self.image_len = self.num_patches**2

    def mlm_ids(self, ori_input_ids, ori_bbox_list):

        expandend_input_length = len(ori_input_ids)

        mask_indices = self.random_spans_noise_mask(expandend_input_length)
        labels_mask = ~mask_indices
        
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        
        input_ids = self.filter_input_ids(ori_input_ids, input_ids_sentinel)
        labels = self.filter_input_ids(ori_input_ids, labels_sentinel)
        
        bbox_list = self.filter_input_bbox(ori_bbox_list, input_ids_sentinel)
        decoder_bbox_list = self.filter_input_bbox(ori_bbox_list, labels_sentinel)
        
        labels_loc = self.create_bbox_label(np.array(decoder_bbox_list), np.array(labels))
        
        return input_ids.tolist(), labels.tolist(), bbox_list, decoder_bbox_list, labels_loc

    def create_bbox_label(self, bbox_list, labels):
        
        def parse_bbox(bbox):
            x0, y0, x1, y1 = 1.0, 1.0, 0.0, 0.0
            for item in bbox:
                x0, y0, x1, y1 = min(x0, item[0]), min(y0, item[1]), max(x1, item[2]), max(y1, item[3])
            return [x0, y0, x1, y1]
        
        sentinel_indices = np.where(self.sentinel_range[0] <= labels)
        split_bbox = np.split(bbox_list, sentinel_indices[0][1:])
        bbox_label = list(map(lambda x: parse_bbox(x[1:]), split_bbox))
        return bbox_label
        
    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[0] = mask_indices[0]
        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (self.sentinel_range[1] - sentinel_ids + 1), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        if type(input_ids[0]) is list:
            input_ids_ = []
            sentinel_ids_ = []
            for i in range(len(input_ids)):
                if sentinel_ids[i] > 0:
                    sentinel_ids_ += [sentinel_ids[i]] + (len(input_ids[i])-1) * [-1]
                else:
                    sentinel_ids_ += len(input_ids[i]) * [sentinel_ids[i]]
                input_ids_ += input_ids[i]
            input_ids_ = np.array(input_ids_)
            sentinel_ids_ = np.array(sentinel_ids_)
        else:
            input_ids_ = input_ids
            sentinel_ids_ = sentinel_ids

        input_ids_full = np.where(sentinel_ids_ != 0, sentinel_ids_, input_ids_)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(-1)
        
        return input_ids
    
    
    def filter_input_bbox(self, bbox_list, sentinel_ids):
        """
        Puts sentinel regions on `bbox` by creating new bboxes around these regions.
        """
        
        new_bbox_list = []
        if type(bbox_list[0][0]) is list:
            for i in range(len(sentinel_ids)):
                if sentinel_ids[i] > 0:
                    new_bbox_list.append([0, 0, 0, 0])
                elif sentinel_ids[i] == 0:
                    new_bbox_list += bbox_list[i]
        else:
            for i in range(len(sentinel_ids)):
                if sentinel_ids[i] > 0:
                    new_bbox_list.append([0, 0, 0, 0])
                elif sentinel_ids[i] == 0:
                    new_bbox_list.append(bbox_list[i])
               
        return new_bbox_list

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
    
    def random_masking(self, L, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask_ratio = random.random() * mask_ratio
        
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(L)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        ids_remove = ids_shuffle[len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([L])
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        return mask, ids_restore, ids_remove, ids_keep
    
    
class DataCollatorForLayoutModeling(DataCollatorForSelfSupervised):
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, noise_density=0.15, mean_noise_span_length=3, pad_token_id=0, decoder_start_token_id=0, image_size=1024):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_loc_id = self.tokenizer._loc_extra_ids - 1
        
        self.image_size = image_size
        self.patch_size = 16
        self.num_patches = self.image_size//self.patch_size
        self.image_len = self.num_patches**2

    def __call__(self, ori_input_ids, ori_bbox):
        
        self.sentinel_range = [32100, 32199]
        self.sentinel_length = self.sentinel_range[1] - self.sentinel_range[0] + 1

        input_ids, sample_text_ids, bbox_list, sample_bbox_list, bbox_labels = self.mlm_ids(ori_input_ids, ori_bbox)
        for i in range(len(input_ids)):
            if self.sentinel_range[0] <= input_ids[i] <= self.sentinel_range[1]:
                bbox_list[i] = [0, 0, 0, 0]      
        
        sample_bbox_label = ''
        for i, item in enumerate(bbox_labels[:self.sentinel_length]):
            sample_bbox_label += f'<loc_{int(item[0]*self.max_loc_id)}><loc_{int(item[1]*self.max_loc_id)}><loc_{int(item[2]*self.max_loc_id)}><loc_{int(item[3]*self.max_loc_id)}>'
        sample_bbox_ids = self.tokenizer.encode(sample_bbox_label, add_special_tokens=False)       
        
        task_prefix = 'layout modeling.'
        task_prefix_ids = self.tokenizer.encode(task_prefix, add_special_tokens=False)
        input_ids = task_prefix_ids + input_ids + [1]
        bbox_list = [[0,0,0,0]] * (len(task_prefix_ids)) + bbox_list + [[1.0,1.0,1.0,1.0]]

        sentinel_indices = np.where(self.sentinel_range[0] <= np.array(sample_text_ids))
        split_sample_ids = np.split(np.array(sample_text_ids), sentinel_indices[0][1:])
        labels = []
        index = 0
        j = 0
        while j < len(input_ids):
            if self.sentinel_range[0] <= input_ids[j] <= self.sentinel_range[1]:
                input_ids[j:j+1] = split_sample_ids[index].tolist() + [input_ids[j]+100]
                bbox_list[j:j+1] = (len(split_sample_ids[index]) + 1) * [[0, 0, 0, 0]]
                labels += [input_ids[j]] + sample_bbox_ids[index*4:(index+1)*4]
                index += 1
            j += 1
        labels += [1]
            
        return input_ids, labels, bbox_list
    
    
class DataCollatorForTextAndLayoutReconstruction(DataCollatorForSelfSupervised):
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, noise_density=0.15, mean_noise_span_length=3, pad_token_id=0, decoder_start_token_id=0, image_size=1024):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_loc_id = self.tokenizer._loc_extra_ids - 1
        
        self.image_size = image_size
        self.patch_size = 16
        self.num_patches = self.image_size//self.patch_size
        self.image_len = self.num_patches**2

    def __call__(self, ori_input_ids, ori_bbox):
        
        self.sentinel_range = [32000, 32099]
        self.sentinel_length = self.sentinel_range[1] - self.sentinel_range[0] + 1
    
        input_ids, sample_text_ids, bbox_list, sample_bbox_list, bbox_labels = self.mlm_ids(ori_input_ids, ori_bbox)
        for i in range(len(input_ids)):
            if self.sentinel_range[0] <= input_ids[i] <= self.sentinel_range[1]:
                bbox_list[i] = [0, 0, 0, 0]      
        
        sample_bbox_label = ''
        for i, item in enumerate(bbox_labels[:self.sentinel_length]):
            sample_bbox_label += f'<loc_{int(item[0]*self.max_loc_id)}><loc_{int(item[1]*self.max_loc_id)}><loc_{int(item[2]*self.max_loc_id)}><loc_{int(item[3]*self.max_loc_id)}>'
        sample_bbox_ids = self.tokenizer.encode(sample_bbox_label, add_special_tokens=False)       

            
        task_prefix = 'joint text and layout reconstruction.'
        task_prefix_ids = self.tokenizer.encode(task_prefix, add_special_tokens=False)
        input_ids = task_prefix_ids + input_ids + [1]
        bbox_list = [[0,0,0,0]] * (len(task_prefix_ids)) + bbox_list + [[1.0,1.0,1.0,1.0]]
        sample_text_ids = sample_text_ids + [1]

        index = 0
        j = 0
        while j < len(sample_text_ids):
            if 1 < sample_text_ids[j] < self.sentinel_range[0]:
                if self.sentinel_range[0] <= sample_text_ids[j+1] <= self.sentinel_range[1] or sample_text_ids[j+1] == 1:
                    sample_text_ids[j+1:j+1] = [3] + sample_bbox_ids[index*4:(index+1)*4]
                    index += 1
            j += 1
        labels = sample_text_ids
            
        return input_ids, labels, bbox_list
    
    
class DataCollatorForVisualTextRecognition(DataCollatorForSelfSupervised):
    """
    Data collator used for T5 document classification
    """
    def __init__(self, tokenizer=None, noise_density=0.15, mean_noise_span_length=3, pad_token_id=0, decoder_start_token_id=0, image_size=1024):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_loc_id = self.tokenizer._loc_extra_ids - 1
        
        self.image_size = image_size
        self.patch_size = 16
        self.num_patches = self.image_size//self.patch_size
        self.image_len = self.num_patches**2

    def __call__(self, ori_input_ids, ori_bbox):
        
        self.sentinel_range = [32300, 32399]
        self.sentinel_length = self.sentinel_range[1] - self.sentinel_range[0] + 1
   
        input_ids, sample_text_ids, bbox_list, sample_bbox_list, bbox_labels = self.mlm_ids(ori_input_ids, ori_bbox)
        for i in range(len(input_ids)):
            if self.sentinel_range[0] <= input_ids[i] <= self.sentinel_range[1]:
                bbox_list[i] = [0, 0, 0, 0]      
        
        sample_bbox_label = ''
        for i, item in enumerate(bbox_labels[:self.sentinel_length]):
            sample_bbox_label += f'<loc_{int(item[0]*self.max_loc_id)}><loc_{int(item[1]*self.max_loc_id)}><loc_{int(item[2]*self.max_loc_id)}><loc_{int(item[3]*self.max_loc_id)}>'
        sample_bbox_ids = self.tokenizer.encode(sample_bbox_label, add_special_tokens=False)       

        task_prefix = 'visual text recognition.'
        task_prefix_ids = self.tokenizer.encode(task_prefix, add_special_tokens=False)
        input_ids = task_prefix_ids + input_ids + [1]
        bbox_list = [[0,0,0,0]] * (len(task_prefix_ids)) + bbox_list + [[1.0,1.0,1.0,1.0]]
        index = 0
        j = 0
        while j < len(input_ids):
            if self.sentinel_range[0] <= input_ids[j] <= self.sentinel_range[1]:
                input_ids[j+1:j+1] = sample_bbox_ids[index*4:(index+1)*4] + [input_ids[j] + 100]
                bbox_list[j+1:j+1] = [[0,0,0,0]] * 5
                index += 1
            j += 1
        labels = sample_text_ids + [1]
            
        return input_ids, labels, bbox_list
