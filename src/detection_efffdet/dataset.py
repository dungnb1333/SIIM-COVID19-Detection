import os
import numpy as np 
import cv2
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SiimCovidDataset(Dataset):
    def __init__(self, df, images_dir, image_size):
        super(SiimCovidDataset,self).__init__()
        self.df = df
        self.images_dir = images_dir
        self.image_size = image_size
        self._transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = '{}/{}.png'.format(self.images_dir, self.df.loc[index, 'imageid'])
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        target = dict(img_idx=index, img_size=(width, height))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t



class SiimCovidExtDataset(Dataset):
    def __init__(self, df, image_size):
        super(SiimCovidExtDataset,self).__init__()
        self.df = df
        self.image_size = image_size
        self._transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = '{}'.format(self.df.loc[index, 'image_path'])
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        target = dict(img_idx=index, img_size=(width, height))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t
