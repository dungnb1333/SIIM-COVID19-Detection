""" COCO, VOC, OpenImages dataset configurations

Copyright 2020 Ross Wightman
"""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CocoCfg:
    variant: str = None
    parser: str = 'coco'
    num_classes: int = 80
    splits: Dict[str, dict] = None


@dataclass
class Coco2017Cfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='annotations/instances_train2017.json', img_dir='train2017', has_labels=True),
        val=dict(ann_filename='annotations/instances_val2017.json', img_dir='val2017', has_labels=True),
        test=dict(ann_filename='annotations/image_info_test2017.json', img_dir='test2017', has_labels=False),
        testdev=dict(ann_filename='annotations/image_info_test-dev2017.json', img_dir='test2017', has_labels=False),
    ))


@dataclass
class Coco2014Cfg(CocoCfg):
    variant: str = '2014'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='annotations/instances_train2014.json', img_dir='train2014', has_labels=True),
        val=dict(ann_filename='annotations/instances_val2014.json', img_dir='val2014', has_labels=True),
        test=dict(ann_filename='', img_dir='test2014', has_labels=False),
    ))


@dataclass
class VocCfg:
    variant: str = None
    parser: str = 'voc'
    num_classes: int = 80
    img_filename: str = '%s.jpg'
    splits: Dict[str, dict] = None


@dataclass
class Voc2007Cfg(VocCfg):
    variant: str = '2007'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename='VOC2007/ImageSets/Main/train.txt',
            ann_filename='VOC2007/Annotations/%s.xml',
            img_dir='VOC2007/JPEGImages', ),
        val=dict(
            split_filename='VOC2007/ImageSets/Main/val.txt',
            ann_filename='VOC2007/Annotations/%s.xml',
            img_dir='VOC2007/JPEGImages'),
        #test=dict(img_dir='JPEGImages')
    ))


@dataclass
class Voc2012Cfg(VocCfg):
    variant: str = '2012'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename='VOC2012/ImageSets/Main/train.txt',
            ann_filename='VOC2012/Annotations/%s.xml',
            img_dir='VOC2012/JPEGImages'),
        val=dict(
            split_filename='VOC2012/ImageSets/Main/val.txt',
            ann_filename='VOC2012/Annotations/%s.xml',
            img_dir='VOC2012/JPEGImages'),
        #test=dict(img_dir='JPEGImages', split_file=None)
    ))


@dataclass
class Voc0712Cfg(VocCfg):
    variant: str = '0712'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['VOC2007/ImageSets/Main/trainval.txt', 'VOC2012/ImageSets/Main/trainval.txt'],
            ann_filename=['VOC2007/Annotations/%s.xml', 'VOC2012/Annotations/%s.xml'],
            img_dir=['VOC2007/JPEGImages', 'VOC2012/JPEGImages']),
        val=dict(
            split_filename='VOC2007/ImageSets/Main/test.txt',
            ann_filename='VOC2007/Annotations/%s.xml',
            img_dir='VOC2007/JPEGImages'),
        #test=dict(img_dir='JPEGImages', split_file=None)
    ))

@dataclass
class RSNAPneuCfg:
    variant: str = None
    parser: str = 'rsnapneu'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['external_dataset/ext_csv/rsna_pneumonia_effdet_train.txt'],
            ann_filename=['external_dataset/rsna-pneumonia-detection-challenge/labels/train/%s.xml'],
            img_dir=['external_dataset/rsna-pneumonia-detection-challenge/images/train']),
        val=dict(
            split_filename=['external_dataset/ext_csv/rsna_pneumonia_effdet_valid.txt'],
            ann_filename=['external_dataset/rsna-pneumonia-detection-challenge/labels/train/%s.xml'],
            img_dir=['external_dataset/rsna-pneumonia-detection-challenge/images/train'])
    ))

@dataclass
class WarmupCfg:
    variant: str = None
    parser: str = 'warmup'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=[
                'external_dataset/ext_csv/rsna_pneumonia_effdet_train.txt',
                'external_dataset/ext_csv/rsna_pneumonia_effdet_valid.txt',
                'pseudo_det/padchest_effdet_ext_train.txt',
                'pseudo_det/pneumothorax_effdet_ext_train.txt',
                'pseudo_det/vin_effdet_ext_train.txt',
                'pseudo_det/test_effdet_ext_train.txt',
            ],
            ann_filename=[
                'external_dataset/rsna-pneumonia-detection-challenge/labels/train/%s.xml',
                'external_dataset/rsna-pneumonia-detection-challenge/labels/train/%s.xml',
                'external_dataset/padchest/labels/%s.xml',
                'external_dataset/pneumothorax/labels/%s.xml',
                'external_dataset/vinbigdata/labels/%s.xml',
                'siim-covid19-detection/labels/test/%s.xml',
            ],
            img_dir=[
                'external_dataset/rsna-pneumonia-detection-challenge/images/train',
                'external_dataset/rsna-pneumonia-detection-challenge/images/train',
                'external_dataset/padchest/images',
                'external_dataset/pneumothorax/images',
                'external_dataset/vinbigdata/images',
                'siim-covid19-detection/images/test',
            ]),
        val=dict(
            split_filename=[
                'siim-covid19-detection/folds/effdet_train_fold0.txt',
                'siim-covid19-detection/folds/effdet_valid_fold0.txt',
            ],
            ann_filename=[
                'siim-covid19-detection/labels/train/%s.xml',
                'siim-covid19-detection/labels/train/%s.xml'
            ],
            img_dir=[
                'siim-covid19-detection/images/train',
                'siim-covid19-detection/images/train'
            ])
    ))

@dataclass
class XrayCfgFold0:
    variant: str = None
    parser: str = 'xray'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['siim-covid19-detection/folds/effdet_train_fold0.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
        val=dict(
            split_filename=['siim-covid19-detection/folds/effdet_valid_fold0.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
    ))

@dataclass
class XrayCfgFold1:
    variant: str = None
    parser: str = 'xray'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['siim-covid19-detection/folds/effdet_train_fold1.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
        val=dict(
            split_filename=['siim-covid19-detection/folds/effdet_valid_fold1.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
    ))

@dataclass
class XrayCfgFold2:
    variant: str = None
    parser: str = 'xray'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['siim-covid19-detection/folds/effdet_train_fold2.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
        val=dict(
            split_filename=['siim-covid19-detection/folds/effdet_valid_fold2.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
    ))

@dataclass
class XrayCfgFold3:
    variant: str = None
    parser: str = 'xray'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['siim-covid19-detection/folds/effdet_train_fold3.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
        val=dict(
            split_filename=['siim-covid19-detection/folds/effdet_valid_fold3.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
    ))

@dataclass
class XrayCfgFold4:
    variant: str = None
    parser: str = 'xray'
    num_classes: int = 1
    img_filename: str = '%s.png'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            split_filename=['siim-covid19-detection/folds/effdet_train_fold4.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
        val=dict(
            split_filename=['siim-covid19-detection/folds/effdet_valid_fold4.txt'],
            ann_filename=['siim-covid19-detection/labels/train/%s.xml'],
            img_dir=['siim-covid19-detection/images/train']),
    ))

@dataclass
class OpenImagesCfg:
    variant: str = None
    parser: str = 'openimages'
    num_classes: int = None
    img_filename = '%s.jpg'
    splits: Dict[str, dict] = None


@dataclass
class OpenImagesObjCfg(OpenImagesCfg):
    num_classes: int = 601
    categories_map: str = 'annotations/class-descriptions-boxable.csv'


@dataclass
class OpenImagesSegCfg(OpenImagesCfg):
    num_classes: int = 350
    categories_map: str = 'annotations/classes-segmentation.txt'


@dataclass
class OpenImagesObjV5Cfg(OpenImagesObjCfg):
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            img_dir='train', img_info='annotations/train-info.csv', has_labels=True, prefix_levels=1,
            ann_bbox='annotations/train-annotations-bbox.csv',
            ann_img_label='annotations/train-annotations-human-imagelabels-boxable.csv',
        ),
        val=dict(
            img_dir='validation', img_info='annotations/validation-info.csv', has_labels=True, prefix_levels=0,
            ann_bbox='annotations/validation-annotations-bbox.csv',
            ann_img_label='annotations/validation-annotations-human-imagelabels-boxable.csv',
        ),
        test=dict(
            img_dir='test', img_info='', has_labels=True, prefix_levels=0,
            ann_bbox='annotations/test-annotations-bbox.csv',
            ann_img_label='annotations/test-annotations-human-imagelabels-boxable.csv',
        )
    ))


@dataclass
class OpenImagesObjChallenge2019Cfg(OpenImagesObjCfg):
    num_classes: int = 500
    categories_map: str = 'annotations/challenge-2019/challenge-2019-classes-description-500.csv'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(
            img_dir='train', img_info='annotations/train-info.csv', has_labels=True, prefix_levels=1,
            ann_bbox='annotations/challenge-2019/challenge-2019-train-detection-bbox.csv',
            ann_img_label='annotations/challenge-2019/challenge-2019-train-detection-human-imagelabels.csv',
        ),
        val=dict(
            img_dir='validation', img_info='annotations/validation-info.csv', has_labels=True, prefix_levels=0,
            ann_bbox='annotations/challenge-2019/challenge-2019-validation-detection-bbox.csv',
            ann_img_label='annotations/challenge-2019/challenge-2019-validation-detection-human-imagelabels.csv',
        ),
        test=dict(
            img_dir='challenge2019', img_info='annotations/challenge-2019/challenge2019-info', prefix_levels=0,
            has_labels=False, ann_bbox='', ann_img_label='',
        )
    ))


@dataclass
class OpenImagesSegV5Cfg(OpenImagesSegCfg):
    num_classes: int = 300
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(),
        val=dict(),
        test=dict()
    ))


@dataclass
class OpenImagesSegChallenge2019Cfg(OpenImagesSegCfg):
    num_classes: int = 300
    ann_class_map: str = 'annotations/challenge-2019/challenge-2019-classes-description-segmentable.csv'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(),
        val=dict(),
        test=dict()
    ))