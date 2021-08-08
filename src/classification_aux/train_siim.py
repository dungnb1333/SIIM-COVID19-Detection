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
from timm.utils.model_ema import ModelEmaV2
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

from models import SiimCovidAuxModel
from dataset import SiimCovidAuxDataset, classes, rsnapneumonia_classes

from utils import seed_everything, refine_dataframe, get_study_map

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/seresnet152d_512_unet.yaml', type=str)
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument("--patience", default=8, type=int)
parser.add_argument("--weighted", default=True, type=lambda x: (str(x).lower() == "true"))

args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    os.makedirs('checkpoints', exist_ok = True)
    df = pd.read_csv('../../dataset/siim-covid19-detection/train_kfold.csv')
    
    for fold in args.folds:
        valid_df = df.loc[df['fold'] == fold]
        valid_df = refine_dataframe(valid_df)
        
        train_df = df.loc[df['fold'] != fold]
        train_df = refine_dataframe(train_df)
        
        if args.frac != 1:
            print('Quick training')
            train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
            valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)

        train_dataset = SiimCovidAuxDataset(
            df=train_df,
            images_dir='../../dataset/siim-covid19-detection/images/train',
            image_size=cfg['aux_image_size'], mode='train')
        valid_dataset = SiimCovidAuxDataset(
            df=valid_df,
            images_dir='../../dataset/siim-covid19-detection/images/train',
            image_size=cfg['aux_image_size'], mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=cfg['aux_batch_size'], sampler=RandomSampler(train_dataset), num_workers=cfg['workers'], drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg['aux_batch_size'], sampler=SequentialSampler(valid_dataset), num_workers=cfg['workers'], drop_last=False)

        print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SiimCovidAuxModel(
            encoder_name=cfg['encoder_name'],
            encoder_weights=None,
            decoder=cfg['decoder'],
            classes=len(classes),
            in_features=cfg['in_features'],
            decoder_channels=cfg['decoder_channels'],
            encoder_pretrained_path=None,
            encoder_pretrained_num_classes=None,
            model_pretrained_path='rsnapneu_pretrain/{}_{}_{}_rsnapneu.pth'.format(cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder']), 
            model_pretrained_num_classes=len(rsnapneumonia_classes))

        model_ema = ModelEmaV2(model, decay=cfg['model_ema_decay'], device=device)
        model.to(device)
    
        if args.weighted:
            cls_criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([0.2, 0.2, 0.3, 0.3]).to(device), reduction='none')
        else:
            cls_criterion = nn.BCEWithLogitsLoss()

        seg_criterion = DiceLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['aux_init_lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['aux_epochs']-1)

        scaler = torch.cuda.amp.GradScaler()

        LOG = 'checkpoints/{}_{}_{}_aux_fold{}.log'.format(cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'], fold)
        CHECKPOINT = 'checkpoints/{}_{}_{}_aux_fold{}.pth'.format(cfg['encoder_name'], cfg['aux_image_size'], cfg['decoder'], fold)

        ema_val_map_max = 0
        if os.path.isfile(LOG):
            os.remove(LOG)
        log_file = open(LOG, 'a')
        log_file.write('epoch, lr, train_loss, train_iou, ema_val_iou, val_map, ema_val_map\n')
        log_file.close()

        count = 0
        best_epoch = 0

        iou_func = IoU(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None)

        for epoch in range(cfg['aux_epochs']):
            scheduler.step()
            model.train()
            train_loss = []
            train_iou = []

            loop = tqdm(train_loader)
            for images, masks, labels in loop:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
 
                if cfg['aux_mixup']:
                    ### mixup
                    lam = np.random.beta(0.5, 0.5)
                    rand_index = torch.randperm(images.size(0))
                    images = lam * images + (1 - lam) * images[rand_index, :,:,:]
                    labels_a, labels_b = labels, labels[rand_index]
                    masks_a, masks_b = masks, masks[rand_index,:,:,:]
                
                    with torch.cuda.amp.autocast():
                        seg_outputs, cls_outputs = model(images)
                        cls_loss = lam * cls_criterion(cls_outputs, labels_a) + (1 - lam) * cls_criterion(cls_outputs, labels_b)
                        if args.weighted:
                            cls_loss = torch.mean(torch.sum(cls_loss, 1),0)
                        seg_loss = lam * seg_criterion(seg_outputs, masks_a) + (1 - lam) * seg_criterion(seg_outputs, masks_b)
                        loss = cfg['aux_weight']*cls_loss + (1-cfg['aux_weight'])*seg_loss

                        train_iou.append(iou_func(seg_outputs, masks).item())
                        train_loss.append(loss.item())
                else:
                    with torch.cuda.amp.autocast():
                        seg_outputs, cls_outputs = model(images)
                        cls_loss = cls_criterion(cls_outputs, labels)
                        if args.weighted:
                            cls_loss = torch.mean(torch.sum(cls_loss, 1),0)
                        seg_loss = seg_criterion(seg_outputs, masks)
                        loss = cfg['aux_weight']*cls_loss + (1-cfg['aux_weight'])*seg_loss

                        train_iou.append(iou_func(seg_outputs, masks).item())
                        train_loss.append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                model_ema.update(model)

                loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, cfg['aux_epochs']-1, optimizer.param_groups[0]['lr']))
                loop.set_postfix(loss=np.mean(train_loss), iou=np.mean(train_iou))
            train_loss = np.mean(train_loss)
            train_iou = np.mean(train_iou)

            model.eval()
            model_ema.eval()

            cls_preds = []
            cls_ema_preds = []
            imageids = []

            emal_val_iou = 0
            for images, masks, labels, ids in tqdm(valid_loader):
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                imageids.extend(ids)

                with torch.cuda.amp.autocast(), torch.no_grad():
                    _, cls_outputs = model(images)
                    cls_preds.append(torch.sigmoid(cls_outputs).data.cpu().numpy())

                    ema_seg_outputs, ema_cls_outputs = model_ema.module(images)
                    cls_ema_preds.append(torch.sigmoid(ema_cls_outputs).data.cpu().numpy())

                    emal_val_iou += iou_func(ema_seg_outputs, masks).item()*images.size(0)

            cls_preds = np.vstack(cls_preds)
            cls_ema_preds = np.vstack(cls_ema_preds)
            imageids = np.array(imageids)

            pred_dict = dict(zip(imageids, cls_preds))
            ema_pred_dict = dict(zip(imageids, cls_ema_preds))

            val_map = get_study_map(valid_df, pred_dict, stride=0.01)['mAP']
            ema_val_map = get_study_map(valid_df, ema_pred_dict, stride=0.01)['mAP']
            emal_val_iou /= len(valid_loader.dataset)
            
            print('train loss: {:.5f} | train iou: {:.5f} | ema_val_iou: {:.5f} | val_map: {:.5f} | ema_val_map: {:.5f}'.format(train_loss, train_iou, emal_val_iou, val_map, ema_val_map))
            log_file = open(LOG, 'a')
            log_file.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                epoch, optimizer.param_groups[0]['lr'], train_loss, train_iou, emal_val_iou, val_map, ema_val_map))
            log_file.close()

            if ema_val_map > ema_val_map_max:
                print('Ema valid map improved from {:.5f} to {:.5f} saving model to {}'.format(ema_val_map_max, ema_val_map, CHECKPOINT))
                ema_val_map_max = ema_val_map
                best_epoch = epoch
                count = 0
                torch.save(model_ema.module.state_dict(), CHECKPOINT)
            else:
                count += 1
            
            if count > args.patience:
                break
        
        log_file = open(LOG, 'a')
        log_file.write('Best epoch {} | mAP max: {}\n'.format(best_epoch, ema_val_map_max))
        log_file.close()
        print('Best epoch {} | mAP max: {}'.format(best_epoch, ema_val_map_max))