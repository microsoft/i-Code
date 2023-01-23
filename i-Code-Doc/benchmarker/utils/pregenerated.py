import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from pdf2image import convert_from_path

from benchmarker.data.data_converter import FEAT_META
from benchmarker.data.model.feature import Feature
from benchmarker.data.utils import IMG_SIZE_DIVISIBILITY, apply_on_nested_dict

from core.common.utils import img_trans_torchvision
# noinspection PyArgumentList
class PregeneratedDatasetBase(Dataset):

    def __init__(self, im_dir, path, data_epoch = None, mode = 'w+',
                 segment_levels = ('tokens', 'lines'),
                 verbose = True, img_conf = None):

        self.verbose = verbose
        self.segment_levels = segment_levels
        self.img_conf = img_conf
        self.mode = mode
        self.data_epoch = data_epoch

        self.im_dir = im_dir

        self.memfile_path = path / f'epoch_{self.data_epoch}' if data_epoch is not None else path
        self.memfile_path.mkdir(exist_ok=True, parents=True)
        self.metrics_file = self.memfile_path / 'metrics.json'
        self.data: Dict[str, Any] = {}
        self.num_samples: int = 0
        self.seq_len = None
        self.vocab_size = None

    @staticmethod
    def validate_img_conf(img_conf):
        width = img_conf['width']
        max_height = img_conf['max_height']
        divisable_msg = f'should be divisable by {IMG_SIZE_DIVISIBILITY}'
        assert width % IMG_SIZE_DIVISIBILITY == 0, f'Incorect width: {width}, {divisable_msg}'
        assert max_height % IMG_SIZE_DIVISIBILITY == 0, f'Incorect max height size: {max_height}, {divisable_msg}'

    def _create_memmap(self) -> None:
        self.data = {
            'seg_data': {},
            'input_ids': self.get_memmap('input_ids', 'input_ids'),
            'input_masks': self.get_memmap('input_masks', 'input_masks'),
        }

        if 'tokens' in self.segment_levels:
            self.data['seg_data']['tokens'] = {}
            self.data['seg_data']['tokens']['bboxes'] = self.get_memmap('bboxes', 'input_bboxes')
        [self._create_segments(lvl) for lvl in self.segment_levels]

    def get_memmap(self, feature: str, filename: str) -> np.memmap:
        feat = FEAT_META[feature]
        try:
            mmap = np.memmap(filename=self.memfile_path / (filename + '.memmap'),
                             shape=tuple([self.num_samples] +
                                         ([self.seq_len] if feat['wide'] else []) +
                                         feat['dim']),
                             mode=self.mode,
                             dtype=feat['dtype'])
        except Exception as e:
            logging.warning(f'Error during openining {filename} memmap')
            raise e
        return mmap

    def _create_segments(self, lvl: str):
        if lvl == 'tokens':
            return
        elif lvl in ('lines', 'pages'):
            self._create_lines_pages_segment(lvl)
        if lvl == 'pages':
            self._amend_pages_segment(lvl)
        if lvl == 'images':
            self._create_images_segment(lvl)
        if lvl == 'lazyimages':
            self._create_lazyimages_segment(lvl)

    def _create_lines_pages_segment(self, lvl: str):
        self.data['seg_data'][lvl] = {}
        self.data['seg_data'][lvl]['bboxes'] = self.get_memmap('bboxes', lvl + '_bboxes')
        self.data['seg_data'][lvl]['ranges'] = self.get_memmap('ranges', lvl + '_ranges')
        self.data['seg_data'][lvl]['masks'] = self.get_memmap('masks', lvl + '_masks')
        self.data['seg_data'][lvl]['token_map'] = self.get_memmap('token_map', lvl + '_token_ids')

    def _amend_pages_segment(self, lvl: str):
        self.data['seg_data'][lvl]['ordinals'] = self.get_memmap('ordinals', lvl + '_ordinals')
        self.data['seg_data'][lvl]['cardinality'] = self.get_memmap('cardinality', lvl + '_cardinality')

    def _create_images_segment(self, lvl: str):
        self.data['seg_data'][lvl] = {}
        try:
            self.data['seg_data'][lvl]['img_data'] = self.get_memmap('img_data', lvl + '_img_data')
        except FileNotFoundError:
            img_meta = FEAT_META['img_data']
            one_img_shape = tuple([1] + img_meta['dim'])
            img_data_shape = tuple([self.num_samples] + img_meta['dim'])
            one_img = np.full(one_img_shape, dtype=img_meta['dtype'], fill_value=img_meta['default'])
            self.data['seg_data'][lvl]['img_data'] = np.broadcast_to(one_img, img_data_shape)

    def _create_lazyimages_segment(self, lvl: str):
        self.data['seg_data'][lvl] = {}
        try:
            self.data['seg_data'][lvl]['path'] = self.get_memmap('path', f'{lvl}_path')
        except FileNotFoundError:
            img_meta = FEAT_META['path']
            one_sample_shape = tuple([1] + img_meta['dim'])
            data_shape = tuple([self.num_samples] + img_meta['dim'])
            one_sample = np.full(one_sample_shape, dtype=img_meta['dtype'], fill_value=img_meta['default'])
            self.data['seg_data'][lvl]['path'] = np.broadcast_to(one_sample, data_shape)

    def save_metrics(self) -> None:
        with self.metrics_file.open('w') as metrics_file:
            metrics = {
                'num_training_examples': self.num_samples,
                'max_seq_len': self.seq_len,
                'vocab_len': self.vocab_size
            }
            metrics_file.write(json.dumps(metrics))

    def fill_row(self, i: int, features: Feature, data: Dict[str, Any]) -> None:
        # if features is None:
        #     return
        for k, v in data.items():
            if isinstance(v, dict):
                self.fill_row(i, features[k], v)
            else:
                v[i] = features[k]

    def check_for_resize(self, i: int) -> None:
        if i + 1 > self.num_samples:
            self.resize_memmaps(self.num_samples + max(self.num_samples // 5, 1000))

    def flush_memmaps(self):
        """
        Function is flushing all in-memory data to disk
        """
        apply_on_nested_dict(lambda x, k: x.flush(), self.data)

    @staticmethod
    def crop_buffer(memmap):
        """
        Function will look into memmap metadata and crop underlying binary file to required size
        Args:
            memmap: memmap object
        """
        path = memmap.filename
        size = np.product(memmap.shape) * memmap.dtype.itemsize
        tmp_path = Path(f'{path}.tmp')
        shutil.move(path, tmp_path)
        with open(path, 'wb') as f:
            with open(tmp_path, 'rb') as tmp:
                PregeneratedDatasetBase.copy_first_n_bytes(tmp, f, size)
        os.remove(tmp_path)

    @staticmethod
    def copy_first_n_bytes(source: BinaryIO, destination: BinaryIO, num_bytes: int, length=256 * 1024):
        """
        Copy data from file-like object source to file-like object destination until desired number of bytes is reached.

        Args:
            source: source file-like object
            destination: destination file-like object
            num_bytes: number of bytes to store in destination file
            length: size of buffer

        """
        bytes_written = 0
        while 1:
            buf = source.read(length)
            if not buf or bytes_written >= num_bytes:
                break
            destination.write(buf[:num_bytes - bytes_written])
            bytes_written += len(buf)

    def finalize_memmaps(self):
        """
        Function is flushing all in-memory data to disk, and cropping all buffers to required size
        Buffers need to be resized due to `resize_memmap` function behaviour
        which is enlarging memmaps with some additional margin.
        """
        self.flush_memmaps()
        apply_on_nested_dict(lambda x, k: self.crop_buffer(x), self.data)

    def resize_memmaps(self, new_dim: int) -> None:
        self.data = apply_on_nested_dict(
            lambda x, k: np.memmap(filename=x.filename, shape=(new_dim,) + x.shape[1:], mode='r+', dtype=x.dtype),
            self.data,
        )
        if self.verbose:
            logging.info(f'Memmaps were resized to {new_dim}')
        self.num_samples = new_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, item: int) -> Dict[str, Any]:
        try:
            # noinspection PyUnusedLocal
            def convert(x, unused):
                y = x[item]
                return np.array(y) if isinstance(y, np.memmap) else y

            item_dict = apply_on_nested_dict(convert, self.data)
            item_dict = self.add_images(item_dict, os.path.join(self.im_dir, item_dict['doc_id']+'.pdf'))
            return item_dict
        except:
            return self[(item + 1) % len(self)]

    def add_images(self, item_dict, im_path):
        """
        Add image data to sample, using path of the image
        If path is missing it will add dummy image (white page)
        """
        im = convert_from_path(im_path)[0]
        img_lst = img_trans_torchvision(im, self.img_conf['size'])
        item_dict['seg_data']['lazyimages'] = {}
        item_dict['seg_data']['lazyimages']['img_lst'] = img_lst
        return item_dict

    def _get_images(self, impath: Optional[Path], item_dict: Dict[str, Any]) -> List[np.ndarray]:
        imgs = []
        width, height, channels = self.img_conf['width'], self.img_conf['max_height'], self.img_conf['channels']
        if impath:
            imgs.extend(self._read_real_images(impath, item_dict, width, channels))
        else:
            # do not waste memory for empty images and create 1px height image
            imgs.append(self._create_dummy_img(width, height, channels))
        return imgs

    def _read_real_images(self, impath: Path, item_dict: Dict[str, Any], width: int, channels: int) -> List[np.ndarray]:
        mask = item_dict['seg_data']['pages']['masks']
        num_pages = item_dict['seg_data']['pages']['ordinals']
        page_sizes = item_dict['seg_data']['pages']['bboxes']
        page_sizes = page_sizes[mask].tolist()
        page_lst = num_pages[mask].tolist()
        return [self._get_page_img(impath, page_no, channels, page_size, width)
                for page_no, page_size in zip(page_lst, page_sizes)]

    def _get_page_img(self, impath, page_no, channels, page_size, width) -> np.ndarray:
        height = self.img_conf['max_height']
        page_path = impath / f'{page_no}.png'
        if page_path.is_file():
            return self.open_image(page_path)
        else:
            logging.warning(f'Could not find a file {page_path}. Dummy image will be added')
            # height = int(page_size[3] / page_size[2] * width)
            return self._create_dummy_img(width, height, channels)

    def open_image(self, page_path):
        """Opens image and convert it in accordance with config """
        width, channels = self.img_conf['width'], self.img_conf['channels']
        height = self.img_conf['max_height']
        img = Image.open(page_path)

        if img.mode != 'RGB' and channels == 3:
            img = img.convert('RGB')
        if img.mode != 'L' and channels == 1:
            img = img.convert('L')
        new_size = (width, height)
        img_resized = np.array(img.resize(new_size))

        return img_resized

    @staticmethod
    def _create_dummy_img(width: int, height: int, channels: int) -> np.ndarray:
        arr_sz = (height, width, 3) if channels == 3 else (height, width)
        return np.full(arr_sz, 255, dtype=np.uint8)

    def add_imtok_maybe(self, item_dict):
        """
        Add image tokens to sample at the end, last text tokens could be removed
        if there is no room for image tokens
        """
        if self.img_conf is None:
            return item_dict
        imtok_per_width = self.img_conf['imtok_per_width']
        imtok_id = self.img_conf['imtok_id']
        if imtok_per_width == 0:
            return item_dict

        pg_mask = item_dict['seg_data']['pages']['masks']
        pg_sizes = item_dict['seg_data']['pages']['bboxes']
        pg_sizes = pg_sizes[pg_mask].tolist()

        bboxes = []
        cum_height = 0
        for pgs in pg_sizes:
            # given that pages are bound together in the vertical direction we need to add page height
            imtok_bbox = self.get_imtok_bbox(pgs[2], pgs[3], imtok_per_width, cum_height)
            cum_height += pgs[3]
            bboxes.append(imtok_bbox)

        bboxes_arr = np.concatenate(bboxes)

        mask = item_dict['attention_mask']
        imtok_len = bboxes_arr.shape[0]
        fill_len = mask.sum()
        total_len = mask.shape[0]

        start_idx = min(total_len - imtok_len, fill_len)
        end_idx = start_idx + imtok_len
        item_dict['attention_mask'][start_idx:end_idx] = True
        item_dict['input_ids'][start_idx:end_idx] = imtok_id
        item_dict['seg_data']['tokens']['bboxes'][start_idx:end_idx] = bboxes_arr

        return item_dict

    @staticmethod
    def get_imtok_bbox(width, height, imtok_per_width, offset_y):
        bbox_width = width / imtok_per_width
        num_of_rows = max(1, round(height/bbox_width))
        bbox_height = height / num_of_rows

        startx_img = np.mgrid[0:width:bbox_width][:imtok_per_width]
        startx_img = np.tile(startx_img.reshape(imtok_per_width, 1), num_of_rows).flatten()
        endx_img = startx_img + bbox_width
        starty_img = np.mgrid[offset_y:height+offset_y:bbox_height][:num_of_rows]
        starty_img = np.tile(starty_img, imtok_per_width)
        endy_img = starty_img + bbox_height
        bboxes = np.stack((startx_img, starty_img, endx_img, endy_img), axis=1)

        return bboxes

    @classmethod
    def load_from_memmap(cls, path: Path, **kwargs) -> 'PregeneratedDatasetBase':
        return cls(path=path, mode='r', **kwargs)


class PregeneratedCustomDataset(PregeneratedDatasetBase):
    def __init__(
        self,
        im_dir,
        path,
        tokenizer = None,
        input_data = None,
        data_epoch = None,
        mode = 'w+',
        max_seq_length = 512,
        segment_levels = ('tokens', 'lines'),
        additional_memmap_files = ('token_label_ids',),
        verbose = True,
        img_conf = None,
    ):
        super().__init__(im_dir, path, data_epoch, mode, segment_levels, verbose, img_conf)
        self.tokenizer = tokenizer

        if self.mode == 'w+':
            assert input_data is not None
            assert self.tokenizer is not None
            self.vocab_size = len(tokenizer)
            self.seq_len = max_seq_length
            if hasattr(input_data, '__len__'):
                self.num_samples = len(input_data)
            else:
                # if input_data is generator and it is not known what will be size of it
                self.num_samples = 1000
        elif self.mode == 'r':
            assert self.metrics_file.is_file(), f'{self.metrics_file} is missing'
            metrics = json.loads(self.metrics_file.read_text())
            self.num_samples = metrics['num_training_examples']
            self.seq_len = metrics['max_seq_len']

        self._create_memmap()
        for keymap in additional_memmap_files:
            self.data[keymap] = self.get_memmap(keymap, keymap)

        if self.mode == 'w+':
            self.fill_memmap(input_data)
            self.save_metrics()

        if self.verbose:
            logging.info('Loading complete!')

    def fill_memmap(self, data: Sequence[Dict[str, Any]]) -> None:
        span_count = 0
        for features_list in tqdm(data, desc='Pregenerating ', disable=not self.verbose):
            for features in features_list:
                self.check_for_resize(span_count)
                self.fill_row(span_count, features, self.data)
                span_count += 1
        if span_count != self.num_samples:
            self.resize_memmaps(span_count)

