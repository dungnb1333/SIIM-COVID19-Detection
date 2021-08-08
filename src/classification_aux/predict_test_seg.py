import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import cv2
import albumentations as albu
import torch
from torch.utils.data import DataLoader

from models import SiimCovidAuxModel
from dataset import SiimCovidCLSTestDataset, classes
from utils import seed_everything

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/seresnet152d_512_unet.yaml', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--workers", default=16, type=int)
parser.add_argument("--ckpt_dir", default='checkpoints', type=str)
args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    os.makedirs('prediction_mask/public_test/{}_{}_{}'.format(cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder']), exist_ok = True)

    test_df = pd.read_csv('../../dataset/siim-covid19-detection/test_meta.csv')

    models = {}
    for fold in args.folds:
        CHECKPOINT = '{}/{}_{}_{}_aux_fold{}.pth'.format(args.ckpt_dir, cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'], fold)
        models[fold] = SiimCovidAuxModel(
            encoder_name=cfg['encoder_name'],
            encoder_weights=None,
            decoder=cfg['decoder'],
            classes=len(classes),
            in_features=cfg['in_features'],
            decoder_channels=cfg['decoder_channels'],
            encoder_pretrained_path=None,
            encoder_pretrained_num_classes=None,
            model_pretrained_path=None, 
            model_pretrained_num_classes=None,
            test_mode=False).cuda()
        models[fold].load_state_dict(torch.load(CHECKPOINT))
        models[fold].eval()

    test_dataset = SiimCovidCLSTestDataset(
        df=test_df,
        images_dir='../../dataset/siim-covid19-detection/images/test',
        image_size=cfg['aux_image_size'],
        seg=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print('Test size: {}'.format(len(test_loader.dataset)))

    for ids, images, _, heights, widths in tqdm(test_loader):
        images = images.cuda()
    
        heights, widths = heights.data.cpu().numpy().astype(int), widths.data.cpu().numpy().astype(int)

        mask_pred = []
        with torch.cuda.amp.autocast(), torch.no_grad():
            for fold in args.folds:
                seg_out, _ = models[fold](images)
                seg_out = torch.squeeze(seg_out)
                mask_pred.append(seg_out)

                seg_out, _ = models[fold](torch.flip(images, dims=(3,)).contiguous())
                seg_out = torch.squeeze(torch.flip(seg_out, dims=(3,)).contiguous())
                mask_pred.append(seg_out)

                seg_out, _ = models[fold](torch.flip(images, dims=(2,)).contiguous())
                seg_out = torch.squeeze(torch.flip(seg_out, dims=(2,)).contiguous())
                mask_pred.append(seg_out)

                seg_out, _ = models[fold](torch.flip(images, dims=(2,3)).contiguous())
                seg_out = torch.squeeze(torch.flip(seg_out, dims=(2,3)).contiguous())
                mask_pred.append(seg_out)

        mask_pred = torch.stack(mask_pred, -1).mean(-1).data.cpu().numpy()*255.0
        mask_pred = mask_pred.astype(np.uint8)

        for id, mask, height, width in zip(ids, mask_pred, heights, widths):
            transform = albu.Resize(height=height, width=width)
            mask = transform(image=mask)['image']
            mask_path = 'prediction_mask/public_test/{}_{}_{}/{}.png'.format(cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'], id)
            cv2.imwrite(mask_path, mask)
