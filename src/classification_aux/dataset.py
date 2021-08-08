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

        label = torch.FloatTensor(self.df.loc[index, rsnapneumonia_classes])

        height, width = image.shape[0:2]
        mask = np.zeros((height, width), dtype=np.uint8)
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
                mask[y1:y2,x1:x2] = np.ones((y2-y1, x2-x1), dtype=np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)

        return image, mask, label


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
        img_path = '{}/{}.png'.format(self.images_dir, self.df.loc[index, 'imageid'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)

        label = torch.FloatTensor(self.df.loc[index, classes])

        height, width = image.shape[0:2]
        mask = np.zeros((height, width), dtype=np.uint8)
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
                mask[y1:y2,x1:x2] = np.ones((y2-y1, x2-x1), dtype=np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)
    
        if self.mode == 'train':
            return image, mask, label
        else:
            return image, mask, label, self.df.loc[index, 'imageid']
    

class SiimCovidCLSTestDataset(Dataset):
    def __init__(self, df, images_dir, image_size, seg=False, lung_crop=False):
        super(SiimCovidCLSTestDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        self.seg = seg
        self.lung_crop = lung_crop
        if lung_crop:
            lung_pred_path = '../detection_lung_yolov5/predictions/yolov5_lungcrop_test_pred_fold3.pth'
            print('Load lung prediction from {}'.format(lung_pred_path))
            self.lung_crop_dict = torch.load(lung_pred_path)

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
        image = np.stack([image, image, image], axis=-1)

        height, width = image.shape[0:2]
        if self.lung_crop:
            x1, y1, x2, y2 = self.lung_crop_dict[self.df.loc[index, 'imageid']]
            image_center_crop = image[y1:y2, x1:x2, :]
        else:
            new_size = int(0.8*min(height, width))
            x1 = (width - new_size)//2
            y1 = (height - new_size)//2
            image_center_crop = image[y1:y1+new_size, x1:x1+new_size, :]

        image = self.transform(image=image)['image']
        image_center_crop = self.transform(image=image_center_crop)['image']

        if self.seg:
            return self.df.loc[index, 'imageid'], image, image_center_crop, height, width
        else:
            return self.df.loc[index, 'imageid'], image, image_center_crop

class SiimCovidCLSExtTestDataset(Dataset):
    def __init__(self, df, image_size, seg=False):
        super(SiimCovidCLSExtTestDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.seg = seg
        self.transform = albu.Compose([
            albu.Resize(self.image_size, self.image_size),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.loc[index, 'image_path']
        img_file = img_path.split('/')[-1]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)

        height, width = image.shape[0:2]

        if not self.seg:
            new_size = int(0.8*min(height, width))
            x1 = (width - new_size)//2
            y1 = (height - new_size)//2
            image_center_crop = image[y1:y1+new_size, x1:x1+new_size, :]

        image = self.transform(image=image)['image']
        if not self.seg:
            image_center_crop = self.transform(image=image_center_crop)['image']
        if self.seg:
            return img_file, image, height, width
        else:
            return img_path, image, image_center_crop


class SiimCovidAuxPseudoDataset(Dataset):
    def __init__(self, df, images_dir, image_size, mode):
        super(SiimCovidAuxPseudoDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        assert mode in ['train', 'valid']
        self.mode = mode

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
        if self.df.loc[index, 'pseudo'] == True:
            img_path = self.df.loc[index, 'image_path']
        else:
            img_path = '{}/{}.png'.format(self.images_dir, self.df.loc[index, 'imageid'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)

        label = torch.FloatTensor(self.df.loc[index, classes])

        height, width = image.shape[0:2]

        if self.df.loc[index, 'pseudo'] == True:
            mask = cv2.imread(self.df.loc[index, 'mask_path'], cv2.IMREAD_GRAYSCALE)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask.float()
            mask /= 255.0
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
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
                    mask[y1:y2,x1:x2] = np.ones((y2-y1, x2-x1), dtype=np.uint8)

            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            mask = mask.float()
        mask = torch.unsqueeze(mask, 0)
    
        if self.mode == 'train':
            return image, mask, label
        else:
            return image, mask, label, self.df.loc[index, 'imageid']
    

class SiimCovidCLSDemoDataset(Dataset):
    def __init__(self, df, lung_pred_path, images_dir, image_size):
        super(SiimCovidCLSDemoDataset, self).__init__()
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        self.lung_pred_dict = torch.load(lung_pred_path)

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
        image = np.stack([image, image, image], axis=-1)

        x1, y1, x2, y2 = self.lung_pred_dict[self.df.loc[index, 'imageid']]
        image_center_crop = image[y1:y2, x1:x2, :]

        image = self.transform(image=image)['image']
        image_center_crop = self.transform(image=image_center_crop)['image']
        return self.df.loc[index, 'imageid'], image, image_center_crop
