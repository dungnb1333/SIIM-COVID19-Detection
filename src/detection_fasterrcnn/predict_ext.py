import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml

import torch
from torch.utils.data import DataLoader

from models import SiimCovidAuxModel
from dataset import SiimCovidCLSExtTestDataset, classes

from utils import seed_everything, refine_det, collate_fn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/resnet200d.yaml', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--frac", default=1.0, type=float)

args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    os.makedirs('predictions', exist_ok = True)

    models = {}
    for fold in args.folds:
        print('*'*20, 'Fold {}'.format(fold), '*'*20)
        CHECKPOINT = 'checkpoints/{}_{}_fold{}.pth'.format(cfg['backbone_name'], cfg['aux_image_size'], fold)

        model = SiimCovidAuxModel(
            backbone_name=cfg['backbone_name'],
            imagenet_pretrained=False,
            num_classes=len(classes),
            in_features=cfg['in_features'],
            backbone_pretrained_path=None, 
            backbone_pretrained_cls_num_classes=None,
            model_pretrained_path=None,
            model_pretrained_cls_num_classes=None)
        model = model.cuda()
        model.load_state_dict(torch.load(CHECKPOINT))
        model.eval()
        models[fold] = model

    for source in ['padchest', 'pneumothorax','vin']:
        print('*'*20, source, '*'*20)
        if source == 'padchest':
            ext_df = pd.read_csv('../../dataset/external_dataset/ext_csv/padchest.csv', usecols=['image_path'])
        elif source == 'pneumothorax':
            ext_df = pd.read_csv('../../dataset/external_dataset/ext_csv/pneumothorax.csv', usecols=['image_path'])
        elif source == 'vin':
            ext_df = pd.read_csv('../../dataset/external_dataset/ext_csv/vin.csv', usecols=['image_path'])
        else:
            raise ValueError('source !!!')

        if args.frac != 1:
            ext_df = ext_df.sample(frac=args.frac).reset_index(drop=True)
        
        ext_df = ext_df.reset_index(drop=True)

        test_dataset = SiimCovidCLSExtTestDataset(df=ext_df, image_size=cfg['aux_image_size'])
        test_loader = DataLoader(test_dataset, batch_size=cfg['aux_batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=False, drop_last=False, collate_fn=collate_fn)

        predict_dict = {}
        for image_paths, images, targets, heights, widths in tqdm(test_loader):
            hflip_images = torch.stack(images)
            vflip_images = torch.stack(images)
            hflip_images = torch.flip(hflip_images, dims=(3,)).contiguous()
            vflip_images = torch.flip(vflip_images, dims=(2,)).contiguous()
            images = list(image.cuda() for image in images)
            hflip_images = list(image.cuda() for image in hflip_images)
            vflip_images = list(image.cuda() for image in vflip_images)
            
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(), torch.no_grad():
                for fold in args.folds:
                    det_outputs = models[fold](images, targets)
                    hflip_det_outputs = models[fold](hflip_images, targets)
                    vflip_det_outputs = models[fold](vflip_images, targets)
                    for image_path, det, hflip_det, vflip_det, height, width in zip(image_paths, det_outputs, hflip_det_outputs, vflip_det_outputs, heights, widths):
                        if image_path not in list(predict_dict.keys()):
                            predict_dict[image_path] = [[],[],[], width, height]
                            
                        box_pred = det['boxes'].data.cpu().numpy().astype(float)
                        score_pred = det['scores'].data.cpu().numpy()
                        box_pred = box_pred/float(cfg['aux_image_size'])
                        box_pred = box_pred.clip(0,1)
                        label_pred = np.zeros_like(score_pred, dtype=int)
                        box_pred, label_pred, score_pred = refine_det(box_pred, label_pred, score_pred)

                        hflip_box_pred = hflip_det['boxes'].data.cpu().numpy().astype(float)
                        hflip_box_pred = hflip_box_pred/float(cfg['aux_image_size'])
                        hflip_box_pred = hflip_box_pred.clip(0,1)
                        hflip_box_pred[:,[0,2]] = 1 - hflip_box_pred[:,[0,2]]
                        hflip_score_pred = hflip_det['scores'].data.cpu().numpy()
                        hflip_label_pred = np.zeros_like(hflip_score_pred, dtype=int)
                        hflip_box_pred, hflip_label_pred, hflip_score_pred = refine_det(hflip_box_pred, hflip_label_pred, hflip_score_pred)

                        vflip_box_pred = vflip_det['boxes'].data.cpu().numpy().astype(float)
                        vflip_box_pred = vflip_box_pred/float(cfg['aux_image_size'])
                        vflip_box_pred = vflip_box_pred.clip(0,1)
                        vflip_box_pred[:,[1,3]] = 1 - vflip_box_pred[:,[1,3]]
                        vflip_score_pred = vflip_det['scores'].data.cpu().numpy()
                        vflip_label_pred = np.zeros_like(vflip_score_pred, dtype=int)
                        vflip_box_pred, vflip_label_pred, vflip_score_pred = refine_det(vflip_box_pred, vflip_label_pred, vflip_score_pred)

                        predict_dict[image_path][0] += [box_pred, hflip_box_pred, vflip_box_pred]
                        predict_dict[image_path][1] += [score_pred, hflip_score_pred, vflip_score_pred]
                        predict_dict[image_path][2] += [label_pred, hflip_label_pred, vflip_label_pred]

        pred_dict_path = 'predictions/{}_{}_fold{}_{}_pred.pth'.format(cfg['backbone_name'], cfg['aux_image_size'], '_'.join(str(x) for x in args.folds), source)
        torch.save(predict_dict, pred_dict_path)
