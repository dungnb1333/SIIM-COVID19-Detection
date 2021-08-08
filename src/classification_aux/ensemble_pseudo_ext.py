import numpy as np
import os
import pandas as pd
import torch

from utils import seed_everything
from dataset import classes

import warnings
warnings.filterwarnings("ignore")

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    os.makedirs('pseudo_csv', exist_ok = True)
    for source in ['pneumothorax', 'padchest', 'vin']:
        if source == 'padchest':
            test_df = pd.read_csv('../../dataset/external_dataset/ext_csv/padchest.csv')
        elif source == 'pneumothorax':
            test_df = pd.read_csv('../../dataset/external_dataset/ext_csv/pneumothorax.csv')
        elif source == 'vin':
            test_df = pd.read_csv('../../dataset/external_dataset/ext_csv/vin.csv')
        
        eb5_study_pred = torch.load('predictions/timm-efficientnet-b5_512_deeplabv3plus_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']
        eb6_study_pred = torch.load('predictions/timm-efficientnet-b6_448_linknet_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']
        eb7_study_pred = torch.load('predictions/timm-efficientnet-b7_512_unetplusplus_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']
        sr152_study_pred = torch.load('predictions/timm-seresnet152d_320_512_unet_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']

        image_paths = []
        labels = []
        for _, row in test_df.iterrows():
            pred =  0.3*eb5_study_pred[row['image_path']] + \
                    0.2*eb6_study_pred[row['image_path']] + \
                    0.2*eb7_study_pred[row['image_path']] + \
                    0.3*sr152_study_pred[row['image_path']]
            
            image_path = row['image_path']
            assert os.path.isfile(image_path) == True
            image_paths.append(image_path)

            labels.append(pred)
        pseudo_test_df = pd.DataFrame()
        pseudo_test_df['image_path'] = np.array(image_paths)
        pseudo_test_df[classes] = np.array(labels, dtype=float)
        pseudo_test_df['pseudo'] = np.array([True]*len(test_df), dtype=bool)
        pseudo_test_df.to_csv('pseudo_csv/pseudo_{}.csv'.format(source), index=False)