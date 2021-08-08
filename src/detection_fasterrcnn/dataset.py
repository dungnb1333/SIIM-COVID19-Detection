import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

chest14_classes = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'No Finding',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
]

chexpert_classes = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

rsnapneumonia_classes = ['normal', 'pneumonia']

classes = [
    'Negative for Pneumonia',
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance'
]

study_submission_classes = {
    'Negative for Pneumonia': 'negative',
    'Typical Appearance': 'typical',
    'Indeterminate Appearance': 'indeterminate',
    'Atypical Appearance': 'atypical'
}

class ExternalDataset(Dataset):
    def __init__(self, df, images_dir, image_size, mode, classes):
        super(ExternalDataset,self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        assert mode in ['train', 'valid']
        self.mode = mode
        self.classes = classes

        if self.mode == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0, value=0, p=0.25),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.25),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.IAAEmboss(),
                    albu.RandomBrightnessContrast(),            
                ], p=0.25),
                albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = '{}/{}'.format(self.images_dir, self.df.loc[index, 'image_path'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self.transform(image=image)['image']
        label = torch.FloatTensor(self.df.loc[index, self.classes])
        return image, label

class RSNAPneuAuxDataset(Dataset):
    def __init__(self, df, images_dir, image_size, mode):
        super(RSNAPneuAuxDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        assert mode in ['train', 'valid']
        self.mode = mode

        if self.mode == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.Rotate(limit=10, interpolation=1, border_mode=0, value=0, p=0.25),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.25),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.IAAEmboss(),
                    albu.RandomBrightnessContrast(),            
                ], p=0.25),
                albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = '{}/{}'.format(self.images_dir, self.df.loc[index, 'image_path'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)

        height, width = image.shape[0:2]
        boxes = []
        if self.df.loc[index, 'hasbox']:
            arr = self.df.loc[index, 'label'].split(' ')
            nums = len(arr) // 6
            assert nums > 0

            for i in range(nums):
                class_name = arr[6*i]
                assert class_name == 'opacity'
                x1 = int(float(arr[6*i+2]))
                y1 = int(float(arr[6*i+3]))
                x2 = int(float(arr[6*i+4]))
                y2= int(float(arr[6*i+5]))
                
                x1 = min(max(0,x1),width)
                x2 = min(max(0,x2),width)
                y1 = min(max(0,y1),height)
                y2 = min(max(0,y2),height)

                if x1 >= x2 or y1 >= y2:
                    continue
                boxes.append([x1, y1, x2, y2])  
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=float)
            catids = np.ones(boxes.shape[0], dtype=int)
        else:
            boxes = np.array([], dtype=float).reshape(0,4)
            catids = np.array([], dtype=int)
        transformed = self.transform(image=image, bboxes=boxes, category_ids=catids)
        image = transformed["image"]
        boxes = transformed['bboxes']
        boxes = np.array(boxes, dtype=float)
        if boxes.shape[0] > 0:
            target = {}
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)
            target['area'] = torch.as_tensor(area, dtype=torch.float32)
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return image, target


class SiimCovidAuxDataset(Dataset):
    def __init__(self, df, images_dir, image_size, mode):
        super(SiimCovidAuxDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        assert mode in ['train', 'valid']
        self.mode = mode

        if self.mode == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.Rotate(limit=10, interpolation=1, border_mode=0, value=0, p=0.25),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.25),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.IAAEmboss(),
                    albu.RandomBrightnessContrast(),            
                ], p=0.25),
                albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = '{}/{}.png'.format(self.images_dir, self.df.loc[index, 'imageid'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)

        height, width = image.shape[0:2]
        boxes = []
        if self.df.loc[index, 'hasbox']:
            arr = self.df.loc[index, 'label'].split(' ')
            nums = len(arr) // 6
            assert nums > 0

            for i in range(nums):
                class_name = arr[6*i]
                assert class_name == 'opacity'
                x1 = int(float(arr[6*i+2]))
                y1 = int(float(arr[6*i+3]))
                x2 = int(float(arr[6*i+4]))
                y2= int(float(arr[6*i+5]))
                
                x1 = min(max(0,x1),width)
                x2 = min(max(0,x2),width)
                y1 = min(max(0,y1),height)
                y2 = min(max(0,y2),height)

                if x1 >= x2 or y1 >= y2:
                    continue
                boxes.append([x1, y1, x2, y2])  
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=float)
            catids = np.ones(boxes.shape[0], dtype=int)
        else:
            boxes = np.array([], dtype=float).reshape(0,4)
            catids = np.array([], dtype=int)
        transformed = self.transform(image=image, bboxes=boxes, category_ids=catids)
        image = transformed["image"]
        boxes = transformed['bboxes']
        boxes = np.array(boxes, dtype=float)
        if boxes.shape[0] > 0:
            target = {}
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)
            target['area'] = torch.as_tensor(area, dtype=torch.float32)
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return image, target
    

class SiimCovidCLSTestDataset(Dataset):
    def __init__(self, df, images_dir, image_size):
        super(SiimCovidCLSTestDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size

        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = '{}/{}.png'.format(self.images_dir, self.df.loc[index, 'imageid'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[0:2]
        image = np.stack([image, image, image], axis=-1)

        image = self.transform(image=image)['image']
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
            "area": torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }
        
        return self.df.loc[index, 'imageid'], image, target, height, width


class SiimCovidCLSExtTestDataset(Dataset):
    def __init__(self, df, image_size):
        super(SiimCovidCLSExtTestDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = '{}'.format(self.df.loc[index, 'image_path'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[0:2]
        image = np.stack([image, image, image], axis=-1)

        image = self.transform(image=image)['image']
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
            "area": torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }
        
        return self.df.loc[index, 'image_path'], image, target, height, width



class PseudoAuxDataset(Dataset):
    def __init__(self, df, image_size, mode):
        super(PseudoAuxDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        assert mode in ['train', 'valid']
        self.mode = mode

        if self.mode == 'train':
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.transform = albu.Compose([
                albu.RandomResizedCrop(height=self.image_size, width=self.image_size, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
                albu.Rotate(limit=10, interpolation=1, border_mode=0, value=0, p=0.25),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.25),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.IAAEmboss(),
                    albu.RandomBrightnessContrast(),            
                ], p=0.25),
                albu.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ], bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.loc[index, 'image_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)

        height, width = image.shape[0:2]
        boxes = []
        arr = self.df.loc[index, 'label'].split(' ')
        nums = len(arr) // 6
        assert nums > 0

        for i in range(nums):
            class_name = arr[6*i]
            if class_name == 'opacity':
                x1 = int(float(arr[6*i+2]))
                y1 = int(float(arr[6*i+3]))
                x2 = int(float(arr[6*i+4]))
                y2= int(float(arr[6*i+5]))
                
                x1 = min(max(0,x1),width)
                x2 = min(max(0,x2),width)
                y1 = min(max(0,y1),height)
                y2 = min(max(0,y2),height)

                if x1 >= x2 or y1 >= y2:
                    continue
                boxes.append([x1, y1, x2, y2])  
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=float)
            catids = np.ones(boxes.shape[0], dtype=int)
        else:
            boxes = np.array([], dtype=float).reshape(0,4)
            catids = np.array([], dtype=int)
        transformed = self.transform(image=image, bboxes=boxes, category_ids=catids)
        image = transformed["image"]
        boxes = transformed['bboxes']
        boxes = np.array(boxes, dtype=float)
        if boxes.shape[0] > 0:
            target = {}
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)
            target['area'] = torch.as_tensor(area, dtype=torch.float32)
            target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }

        return image, target