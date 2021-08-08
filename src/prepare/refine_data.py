import os
import pandas as pd 
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    ### remove unused file in chexpert dataset
    chextpert_train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chexpert_train.csv')
    chextpert_valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chexpert_valid.csv')
    chextpert_df = pd.concat([chextpert_train_df, chextpert_valid_df], ignore_index=False)
    useful_image_paths = np.unique(chextpert_df.image_path.values)
    image_paths = []
    for rdir, _, files in os.walk('../../dataset/external_dataset/chexpert/train'):
        for file in files:
            image_path = os.path.join(rdir, file)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
    print(len(chextpert_df), len(image_paths))
    for image_path in tqdm(image_paths):
        if image_path not in useful_image_paths:
            os.remove(image_path)
            print('remove {} ...'.format(image_path))
    
    ### remove unused file in chest14 dataset
    chest14_train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chest14_train.csv')
    chest14_valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chest14_valid.csv')
    chest14_df = pd.concat([chest14_train_df, chest14_valid_df], ignore_index=False)
    useful_image_paths = []
    for image_path in np.unique(chest14_df.image_path.values):
        useful_image_paths.append('../../dataset/external_dataset/chest14/images/{}'.format(image_path))
    image_paths = []
    for rdir, _, files in os.walk('../../dataset/external_dataset/chest14/images'):
        for file in files:
            image_path = os.path.join(rdir, file)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
    print(len(chest14_df), len(image_paths))
    for image_path in tqdm(image_paths):
        if image_path not in useful_image_paths:
            os.remove(image_path)
            print('remove {} ...'.format(image_path))
    
    ### remove unused file in padchest dataset
    padchest_df = pd.read_csv('../../dataset/external_dataset/ext_csv/padchest.csv')
    useful_image_paths = np.unique(padchest_df.image_path.values)
    image_paths = []
    for rdir, _, files in os.walk('../../dataset/external_dataset/padchest/images'):
        for file in files:
            image_path = os.path.join(rdir, file)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
    print(len(padchest_df), len(image_paths))
    for image_path in tqdm(image_paths):
        if image_path not in useful_image_paths:
            os.remove(image_path)
            print('remove {} ...'.format(image_path))