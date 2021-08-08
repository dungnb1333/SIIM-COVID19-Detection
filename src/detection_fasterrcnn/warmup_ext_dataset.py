import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from mean_average_precision import MetricBuilder

from models import SiimCovidAuxModel
from dataset import PseudoAuxDataset, SiimCovidAuxDataset, classes
from dataset import chest14_classes

from utils import seed_everything, collate_fn

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/resnet200d.yaml', type=str)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--patience", default=8, type=int)

args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

def make_pseudo_df():
    output = []
    for source in ['padchest', 'pneumothorax', 'vin', 'test']:
        ext_df = pd.read_csv('../../dataset/pseudo_csv_det/{}.csv'.format(source))
        output.append(ext_df)
    pseudo_df = pd.concat(output, ignore_index=True)
    pseudo_df = pseudo_df.loc[pseudo_df['label'] != 'none 1 0 0 1 1'].reset_index(drop=True)
    return pseudo_df

def make_siim_df(in_df):
    image_paths = []
    for _, row in in_df.iterrows():
        image_path = '../../dataset/siim-covid19-detection/images/train/{}.png'.format(row['imageid'])
        image_paths.append(image_path)
    in_df['image_path'] = np.array(image_paths, dtype=str)
    return in_df

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['aux_epochs'] = args.epochs
    print(cfg)

    ckpt_dir = 'warmup'
    os.makedirs(ckpt_dir, exist_ok = True)

    pseudo_df = make_pseudo_df()
    rsna_df = pd.read_csv('../../dataset/external_dataset/ext_csv/rsna_pneumonia.csv')
    rsna_df = rsna_df.loc[rsna_df['label'] != 'none 1 0 0 1 1'].reset_index(drop=True)
    train_df = pd.concat([rsna_df, pseudo_df], ignore_index=True)

    valid_df = pd.read_csv('../../dataset/siim-covid19-detection/train_kfold.csv')
    valid_df = valid_df.loc[valid_df['label'] != 'none 1 0 0 1 1'].reset_index(drop=True)
    valid_df = make_siim_df(valid_df)

    if args.frac != 1:
        print('Quick training')
        train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
        valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)

    train_dataset = PseudoAuxDataset(
        df=train_df,
        image_size=cfg['aux_image_size'], mode='train')
    valid_dataset = SiimCovidAuxDataset(
        df=valid_df,
        images_dir='../../dataset/siim-covid19-detection/images/train',
        image_size=cfg['aux_image_size'], mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=cfg['aux_batch_size'], sampler=RandomSampler(train_dataset), num_workers=cfg['workers'], pin_memory=False, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['aux_batch_size'], sampler=SequentialSampler(valid_dataset), num_workers=cfg['workers'], pin_memory=False, drop_last=False, collate_fn=collate_fn)

    print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = 'pretrain/{}_{}_pretrain_step1.pth'.format(cfg['backbone_name'], cfg['chest14_image_size'])
    pretrained_num_classes = len(chest14_classes)
    
    model = SiimCovidAuxModel(
        backbone_name=cfg['backbone_name'],
        imagenet_pretrained=False,
        num_classes=len(classes),
        in_features=cfg['in_features'],
        backbone_pretrained_path=pretrained_path, 
        backbone_pretrained_cls_num_classes=pretrained_num_classes,
        model_pretrained_path=None, 
        model_pretrained_cls_num_classes=None)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['aux_init_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['aux_epochs']-1)

    scaler = torch.cuda.amp.GradScaler()

    LOG = '{}/{}_{}_warmup.log'.format(ckpt_dir, cfg['backbone_name'], cfg['aux_image_size'])
    CHECKPOINT = '{}/{}_{}_warmup.pth'.format(ckpt_dir, cfg['backbone_name'], cfg['aux_image_size'])

    val_map_max = 0
    if os.path.isfile(LOG):
        os.remove(LOG)
    log_file = open(LOG, 'a')
    log_file.write('epoch, lr, train_loss, val_map\n')
    log_file.close()

    count = 0
    best_epoch = 0

    for epoch in range(cfg['aux_epochs']):
        scheduler.step()
        model.train()
        train_loss = []

        loop = tqdm(train_loader)
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                det_loss_dict = model(images, targets)
                loss = sum(l for l in det_loss_dict.values())
                train_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, cfg['aux_epochs']-1, optimizer.param_groups[0]['lr']))
            loop.set_postfix(loss=np.mean(train_loss))
        train_loss = np.mean(train_loss)
        
        model.eval()
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
        for images, targets in tqdm(valid_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(), torch.no_grad():
                det_outputs = model(images, targets)

                for t, d in zip(targets, det_outputs):
                    gt_boxes = t['boxes'].data.cpu().numpy()
                    gt_boxes = np.hstack((gt_boxes, np.zeros((gt_boxes.shape[0], 3), dtype=gt_boxes.dtype)))
                    
                    det_boxes = d['boxes'].data.cpu().numpy()
                    det_scores = d['scores'].data.cpu().numpy()
                    det_scores = det_scores.reshape(det_scores.shape[0], 1)
                    det_pred = np.hstack((det_boxes, np.zeros((det_boxes.shape[0], 1), dtype=det_boxes.dtype), det_scores))
                    metric_fn.add(det_pred, gt_boxes)
        val_map = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1), mpolicy='soft')['mAP']

        print('train loss: {:.5f} | val_map: {:.5f}'.format(train_loss, val_map))
        log_file = open(LOG, 'a')
        log_file.write('{}, {:.5f}, {:.5f}, {:.5f}\n'.format(
            epoch, optimizer.param_groups[0]['lr'], train_loss, val_map))
        log_file.close()

        if val_map > val_map_max:
            print('Valid map improved from {:.5f} to {:.5f} saving model to {}'.format(val_map_max, val_map, CHECKPOINT))
            val_map_max = val_map
            best_epoch = epoch
            count = 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            count += 1
        
        if count > args.patience:
            break
    
    log_file = open(LOG, 'a')
    log_file.write('Best epoch {} | val loss min: {}\n'.format(best_epoch, val_map_max))
    log_file.close()
    print('Best epoch {} | val loss min: {}'.format(best_epoch, val_map_max))