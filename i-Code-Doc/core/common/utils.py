import os
import re
import logging

import numpy as np
import torch

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)
PREFIX_CHECKPOINT_DIR = 'checkpoint'
_re_checkpoint = re.compile(r'^' + PREFIX_CHECKPOINT_DIR + r'\-(\d+)$')
    
    
def get_visual_bbox(image_size=224):
    image_feature_pool_shape = [image_size//16, image_size//16]
    visual_bbox_x = (torch.arange(
        0,
        1.0 * (image_feature_pool_shape[1] + 1),
        1.0,
    ) / image_feature_pool_shape[1])
    visual_bbox_y = (torch.arange(
        0,
        1.0 * (image_feature_pool_shape[0] + 1),
        1.0,
    ) / image_feature_pool_shape[0])
    visual_bbox_input = torch.stack(
        [
            visual_bbox_x[:-1].repeat(
                image_feature_pool_shape[0], 1),
            visual_bbox_y[:-1].repeat(
                image_feature_pool_shape[1], 1).transpose(
                    0, 1),
            visual_bbox_x[1:].repeat(
                image_feature_pool_shape[0], 1),
            visual_bbox_y[1:].repeat(
                image_feature_pool_shape[1], 1).transpose(
                    0, 1),
        ],
        dim=-1,
    ).view(-1, 4)
    return visual_bbox_input


class Normalize(object):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def __call__(self, image):
        if 'bgr' in self.format:
            image = image[[2, 1, 0]]
        if '255' in self.format:
            image = image * 255
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
    
    
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path for path in content if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints,
            key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def img_trans_torch(image, image_size=224):
    trans = T.Compose([
            T.ToTensor(),
            T.Resize([image_size,image_size]),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    image = trans(image)  # copy to make it writeable
    return image


def img_resize(image, image_size=224):
    trans = T.Compose([
            T.Resize([image_size,image_size]),
        ])

    image = trans(image)  # copy to make it writeable
    return image


def img_trans_torchvision(image, image_size=224):
    trans = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    image = trans(image)  # copy to make it writeable
    return image


def img_trans_torchvision_int(image, image_size=384):
    trans = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    image = trans(image)  # copy to make it writeable
    return image

def load_image(image_path):
    image = Image.open(image_path).resize((224,224)).convert('RGB')
    h, w = image.size
    image = torch.tensor(np.array(image))
    return image, (w, h)

def convert_img_to_numpy(img):
    return np.array(img)

def normalize_bbox(bbox, size, scale=1000):
    return [
        int(clamp((scale * bbox[0] / size[0]), 0, scale)),
        int(clamp((scale * bbox[1] / size[1]), 0, scale)),
        int(clamp((scale * bbox[2] / size[0]), 0, scale)),
        int(clamp((scale * bbox[3] / size[1]), 0, scale))
    ]
