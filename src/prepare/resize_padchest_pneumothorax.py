import numpy as np 
import os 
import cv2
from multiprocessing import Pool
import albumentations as albu
import pandas as pd
transform = albu.LongestMaxSize(max_size=1024, interpolation=1, always_apply=False, p=1)
                
def resize_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if np.max(image.shape[0:2]) > 1024:
        image = transform(image=image)['image']
        cv2.imwrite(image_path, image)
    
if __name__ == '__main__':
    image_paths = []
    for rdir, _, files in os.walk('../../dataset/external_dataset/pneumothorax/images'):
        for file in files:
            image_path = os.path.join(rdir, file)
            image_paths.append(image_path)
    for rdir, _, files in os.walk('../../dataset/external_dataset/padchest/images'):
        for file in files:
            image_path = os.path.join(rdir, file)
            image_paths.append(image_path)
    
    print(len(image_paths))
    p = Pool(16)
    results = p.map(func=resize_image, iterable = image_paths)
    p.close()
