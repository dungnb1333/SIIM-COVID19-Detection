import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

from models import SiimCovidAuxModel
from dataset import SiimCovidCLSTestDataset, classes
from utils import seed_everything, get_study_map

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/seresnet152d_512_unet.yaml', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--workers", default=16, type=int)
parser.add_argument("--num_tta", default=8, choices=[0,4,6,8], type=int)
parser.add_argument("--ckpt_dir", default='checkpoints_v4', type=str)
args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    df = pd.read_csv('../../dataset/siim-covid19-detection/train_kfold.csv')
    model = SiimCovidAuxModel(
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
        test_mode=True
    )
    
    model = model.cuda()
    model.eval()

    oof_val_pred_dict = {}
    oof_df = []

    for fold in args.folds:
        print('*'*20, 'Fold {}'.format(fold), '*'*20)

        CHECKPOINT = '{}/{}_{}_{}_aux_fold{}.pth'.format(args.ckpt_dir, cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'], fold)
        model.load_state_dict(torch.load(CHECKPOINT))

        valid_df = df.loc[df['fold'] == fold]

        if args.frac != 1:
            valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)
        
        valid_dataset = SiimCovidCLSTestDataset(
            df=valid_df,
            images_dir='../../dataset/siim-covid19-detection/images/train',
            image_size=cfg['aux_image_size'])

        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        print('Valid size: {}'.format(len(valid_loader.dataset)))

        preds = []
        imageids = []
        for ids, images, images_center_crop in tqdm(valid_loader):
            images = images.cuda()
            images_center_crop = images_center_crop.cuda()
            imageids.extend(ids)

            pred = []
            with torch.cuda.amp.autocast(), torch.no_grad():
                pred.append(torch.sigmoid(model(images)))
                if args.num_tta >= 4:
                    pred.append(torch.sigmoid(model(torch.flip(images, dims=(3,)).contiguous())))
                    pred.append(torch.sigmoid(model(torch.flip(images, dims=(2,)).contiguous())))
                    pred.append(torch.sigmoid(model(images_center_crop)))
                
                if args.num_tta >= 6:
                    pred.append(torch.sigmoid(model(torch.flip(images_center_crop, dims=(3,)).contiguous())))
                    pred.append(torch.sigmoid(model(torch.flip(images_center_crop, dims=(2,)).contiguous())))
                
                if args.num_tta >= 8:
                    pred.append(torch.sigmoid(model(torch.flip(images, dims=(2,3)).contiguous())))
                    pred.append(torch.sigmoid(model(torch.flip(images_center_crop, dims=(2,3)).contiguous())))

            pred = torch.mean(torch.stack(pred, -1),-1).data.cpu().numpy()
            preds.append(pred)
            del pred

        preds = np.concatenate(preds, axis=0)
        imageids = np.array(imageids)

        pred_dict = dict(zip(imageids, preds))

        oof_df.append(valid_df)
        oof_val_pred_dict.update(pred_dict)

    oof_df = pd.concat(oof_df, ignore_index=True)

    print('*'*20, 'OOF', '*'*20)
    oof_output = get_study_map(oof_df, oof_val_pred_dict, num_classes=4, stride=0.01)
    print(oof_output)