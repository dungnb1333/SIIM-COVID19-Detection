import argparse
import numpy as np
import os
import pandas as pd
import yaml
import cv2
import torch

from multiprocessing import Pool
from utils import seed_everything
from dataset import classes

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfgs", default=[
    'configs/eb5_512_deeplabv3plus.yaml',
    'configs/eb6_448_linknet.yaml',
    'configs/eb7_512_unetplusplus.yaml',
    'configs/seresnet152d_512_unet.yaml'], nargs="+", type=str)
args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

class ME:
    def __init__(self, imageid, mask_paths):
        self.imageid = imageid
        self.mask_paths = mask_paths

def ensemble(ele):
    masks = []
    for mask_path in ele.mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    masks = np.array(masks, dtype=np.float32)
    masks = np.mean(masks, 0).astype(np.uint8)
    mask_path = 'prediction_mask/public_test/masks/{}.png'.format(ele.imageid)
    cv2.imwrite(mask_path, masks)
    
if __name__ == "__main__":
    os.makedirs('prediction_mask/public_test/masks', exist_ok = True)
    os.makedirs('pseudo_csv', exist_ok = True)

    test_df = pd.read_csv('../../dataset/siim-covid19-detection/test_meta.csv')
    meles = []
    for imageid in np.unique(test_df.imageid.values):
        mask_paths = []
        for cfg_path in args.cfgs:
            with open(cfg_path) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            mask_path = 'prediction_mask/public_test/{}_{}_{}/{}.png'.format(cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'], imageid)
            assert os.path.isfile(mask_path)
            mask_paths.append(mask_path)
        meles.append(ME(imageid, mask_paths))
    
    p = Pool(16)
    results = p.map(func=ensemble, iterable = meles)
    p.close()

    eb5_study_pred = torch.load('predictions/timm-efficientnet-b5_512_deeplabv3plus_aux_fold0_1_2_3_4_test_pred_8tta.pth')['pred_dict']
    eb6_study_pred = torch.load('predictions/timm-efficientnet-b6_448_linknet_aux_fold0_1_2_3_4_test_pred_8tta.pth')['pred_dict']
    eb7_study_pred = torch.load('predictions/timm-efficientnet-b7_512_unetplusplus_aux_fold0_1_2_3_4_test_pred_8tta.pth')['pred_dict']
    sr152_study_pred = torch.load('predictions/timm-seresnet152d_320_512_unet_aux_fold0_1_2_3_4_test_pred_8tta.pth')['pred_dict']

    image_paths = []
    mask_paths = []
    labels = []
    for _, row in test_df.iterrows():
        pred =  0.3*eb5_study_pred[row['imageid']] + \
                0.2*eb6_study_pred[row['imageid']] + \
                0.2*eb7_study_pred[row['imageid']] + \
                0.3*sr152_study_pred[row['imageid']]
        

        image_path = '../../dataset/siim-covid19-detection/images/test/{}.png'.format(row['imageid'])
        assert os.path.isfile(image_path) == True
        image_paths.append(image_path)
        mask_path = 'prediction_mask/public_test/masks/{}.png'.format(row['imageid'])
        assert os.path.isfile(mask_path) == True
        mask_paths.append(mask_path)
        labels.append(pred)
    pseudo_test_df = pd.DataFrame()
    pseudo_test_df['image_path'] = np.array(image_paths)
    pseudo_test_df['mask_path'] = np.array(mask_paths)
    pseudo_test_df[classes] = np.array(labels, dtype=float)
    pseudo_test_df['pseudo'] = np.array([True]*len(test_df), dtype=bool)
    pseudo_test_df.to_csv('pseudo_csv/pseudo_public_test.csv', index=False)