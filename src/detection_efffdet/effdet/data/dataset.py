""" Detection dataset

Hacked together by Ross Wightman
"""
import torch.utils.data as data
import numpy as np
import random
import albumentations as albu

from PIL import Image
from .parsers import create_parser

class XrayDetectionDatset(data.Dataset):
    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None, split=None):
        super(XrayDetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform
        self.split = split

        self.albu_transform = albu.Compose([
            albu.Rotate(limit=5, interpolation=1, border_mode=0, value=0, p=0.25),
            albu.OneOf([
                albu.IAAAdditiveGaussianNoise(p=0.1),
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.RandomContrast(p=1),
            ],p=0.75),
            albu.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.5),
        ])

    def __getitem__(self, index):
        img, target = self.load_image(index)

        if self.split == 'train':
            if random.random() > 0.25:
                while True:
                    mix_index = random.randint(0, len(self)-1)
                    if self._parser.img_infos[index]['id'] != self._parser.img_infos[mix_index]['id']:
                        break
                mix_img, mix_target = self.load_image(mix_index)

                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + mix_img * (1 - r)).astype(np.uint8)
                target['bbox'] = np.vstack([target['bbox'], mix_target['bbox']])
                target['cls'] = np.concatenate([target['cls'], mix_target['cls']])
                
                if target['img_scale'] > mix_target['img_scale']:
                    target['img_size'] = mix_target['img_size']

        return img, target


    def load_image(self, index):
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')

        if self.split == 'train':
            img = np.asarray(img)
            img = self.albu_transform(image=img)['image']
            img = Image.fromarray(img)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None, split=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform
        self.split = split

    def __getitem__(self, index):
        img, target = self.load_image(index)
        
        if self.split == 'train':
            if random.random() > 0.5:
                while True:
                    mix_index = random.randint(0, len(self)-1)
                    if self._parser.img_infos[index]['id'] != self._parser.img_infos[mix_index]['id']:
                        break
                mix_img, mix_target = self.load_image(mix_index)

                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                
                img = (img * r + mix_img * (1 - r)).astype(np.uint8)
                target['bbox'] = np.vstack([target['bbox'], mix_target['bbox']])
                target['cls'] = np.concatenate([target['cls'], mix_target['cls']])

                if target['img_scale'] > mix_target['img_scale']:
                    target['img_size'] = mix_target['img_size']
                    target['img_scale'] = mix_target['img_scale']

                del mix_img
                del mix_target
    
        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    def load_image(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class SkipSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        n (int): skip rate (select every nth)
    """
    def __init__(self, dataset, n=2):
        self.dataset = dataset
        assert n >= 1
        self.indices = np.arange(len(dataset))[::n]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def parser(self):
        return self.dataset.parser

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        self.dataset.transform = t
