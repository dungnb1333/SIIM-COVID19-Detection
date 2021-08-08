import random
import os
import numpy as np
import torch
import pandas as pd 
import pickle

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def collate_fn(batch):
    return tuple(zip(*batch))

def refine_dataframe(in_df):
    out_df = []
    for studyid, grp in in_df.groupby('studyid'):
        if len(grp) > 1:
            check = False
            for _, row in grp.iterrows():
                if row['hasbox']:
                    check = True
            if check:
                tmp_df = grp.loc[grp['hasbox']==True]
                out_df.append(tmp_df)
            else:
                out_df.append(grp)
        else:
            out_df.append(grp)

    out_df = pd.concat(out_df, ignore_index=True)
    return out_df

def refine_det(boxes, labels, scores):
    boxes = boxes.clip(0,1)

    boxes_out = []
    labels_out = []
    scores_out = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        if x1==x2 or y1==y2:
            continue
        box = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
        boxes_out.append(box)
        labels_out.append(label)
        scores_out.append(score)
    return boxes_out, labels_out, scores_out

def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)