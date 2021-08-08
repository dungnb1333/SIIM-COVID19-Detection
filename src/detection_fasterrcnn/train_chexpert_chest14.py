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
from warmup_scheduler import GradualWarmupScheduler
from models import PretrainModel
from dataset import ExternalDataset, chexpert_classes, chest14_classes
from utils import seed_everything

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default='configs/resnet200d.yaml', type=str)
parser.add_argument("--steps", default=[0,1], nargs="+", type=int)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument("--warmup-factor", default=10, type=int)
args = parser.parse_args()
print(args)

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)

    ckpt_dir = 'pretrain'
    os.makedirs(ckpt_dir, exist_ok = True)
    for step in args.steps:
        if step == 0:
            print('Train chexpert')
            train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chexpert_train.csv')
            valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chexpert_valid.csv')
            images_dir='.'
            batch_size = cfg['chexpert_batch_size']
            image_size = cfg['chexpert_image_size']
            dst_classes = chexpert_classes
            init_lr = cfg['chexpert_init_lr']
            epochs = cfg['chexpert_epochs']
            imagenet_pretrained = True
            pretrained_path = None
            pretrained_num_classes = None
        elif step == 1:
            print('Train chest14')
            train_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chest14_train.csv')
            valid_df = pd.read_csv('../../dataset/external_dataset/ext_csv/chest14_valid.csv')
            images_dir = '../../dataset/external_dataset/chest14/images'
            batch_size = cfg['chest14_batch_size']
            image_size = cfg['chest14_image_size']
            dst_classes = chest14_classes
            init_lr = cfg['chest14_init_lr']
            epochs = cfg['chest14_epochs']
            imagenet_pretrained = False
            pretrained_path = '{}/{}_{}_pretrain_step{}.pth'.format(ckpt_dir, cfg['backbone_name'], cfg['chexpert_image_size'], step-1)
            pretrained_num_classes = len(chexpert_classes)
        
        batch_size *= torch.cuda.device_count()
        
        if args.frac != 1:
            print('Quick training')
            train_df = train_df.sample(frac=args.frac).reset_index(drop=True)
            valid_df = valid_df.sample(frac=args.frac).reset_index(drop=True)

        train_dataset = ExternalDataset(df=train_df,images_dir=images_dir,image_size=image_size, mode='train',classes=dst_classes)
        valid_dataset = ExternalDataset(df=valid_df,images_dir=images_dir,image_size=image_size, mode='valid',classes=dst_classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset), num_workers=cfg['workers'])
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset), num_workers=cfg['workers'])

        print('TRAIN: {} | VALID: {}'.format(len(train_loader.dataset), len(valid_loader.dataset)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = PretrainModel(
            backbone_name=cfg['backbone_name'],
            imagenet_pretrained=imagenet_pretrained,
            num_classes=len(dst_classes),
            in_features=cfg['in_features'], 
            backbone_pretrained_path=pretrained_path, 
            backbone_pretrained_num_classes=pretrained_num_classes)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        if step == 0:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-1)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=args.warmup_factor, total_epoch=1, after_scheduler=scheduler_cosine)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-1)

        scaler = torch.cuda.amp.GradScaler()

        LOG = '{}/{}_{}_pretrain_step{}.log'.format(ckpt_dir, cfg['backbone_name'], image_size, step)
        CHECKPOINT = '{}/{}_{}_pretrain_step{}.pth'.format(ckpt_dir, cfg['backbone_name'], image_size, step)
        val_loss_min = np.Inf
        if os.path.isfile(LOG):
            os.remove(LOG)
        log_file = open(LOG, 'a')
        log_file.write('epoch, lr, train_loss, val_loss\n')
        log_file.close()

        for epoch in range(epochs):
            scheduler.step()
            model.train()
            train_loss = []

            loop = tqdm(train_loader)
            for images, labels in loop:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(): 
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss.append(loss.item())
                loop.set_description('Epoch {:02d}/{:02d} | LR: {:.5f}'.format(epoch, epochs-1, optimizer.param_groups[0]['lr']))
                loop.set_postfix(loss=np.mean(train_loss))
            train_loss = np.mean(train_loss)

            model.eval()

            val_loss = 0.0
            for images, labels in tqdm(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast(), torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs.float(), labels)
                
                val_loss += loss.item() * images.size(0)
            val_loss = val_loss / len(valid_loader.dataset)
            
            print('train loss: {:.5f} | val_loss: {:.5f}'.format(train_loss, val_loss))
            log_file = open(LOG, 'a')
            log_file.write('{}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                epoch, optimizer.param_groups[0]['lr'], 
                train_loss, val_loss))
            log_file.close()

            if val_loss < val_loss_min:
                print('Valid loss improved from {:.5f} to {:.5f} saving model to {}'.format(val_loss_min, val_loss, CHECKPOINT))
                val_loss_min = val_loss
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), CHECKPOINT)
                else:
                    torch.save(model.state_dict(), CHECKPOINT)

        del model
        del optimizer
        torch.cuda.empty_cache()