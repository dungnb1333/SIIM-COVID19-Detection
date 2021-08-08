import random
import os
import numpy as np
import torch
import pandas as pd 
from mean_average_precision import MetricBuilder

from dataset import classes

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def get_study_map(df, pred_dict, num_classes=6, stride=0.1):
    assert num_classes in [4,6]
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)

    ### Study level ###
    for studyid, grp in df.groupby('studyid'):
        gts = []
        for clsidx, clsname in enumerate(classes):
            assert len(np.unique(grp[clsname].values)) == 1
            if grp[clsname].values[0] == 1:
                gts.append([0, 0, 1, 1, clsidx, 0, 0])
        gts = np.array(gts)

        study_preds = []
        for _, row in grp.iterrows():
            study_preds.append(pred_dict[row['imageid']])
        study_preds = np.array(study_preds)
        study_preds = np.mean(study_preds, axis=0)
            
        preds = []
        for clsidx in range(len(classes)):
            preds.append([0, 0, 1, 1, clsidx, study_preds[clsidx]])
        preds = np.array(preds)

        metric_fn.add(preds, gts)

    ### Image level ###
    if num_classes == 6:
        for _, row in df.iterrows():
            gts = []
            arr = row['label'].split(' ')
            nums = len(arr) // 6
            for i in range(nums):
                class_name = arr[6*i]
                conf = int(arr[6*i+1])
                if class_name == 'opacity':
                    clsid = 5
                else:
                    clsid = 4
                x1 = int(float(arr[6*i+2]))
                y1 = int(float(arr[6*i+3]))
                x2 = int(float(arr[6*i+4]))
                y2= int(float(arr[6*i+5]))
                gts.append([x1, y1, x2, y2, clsid, 0, 0])
            gts = np.array(gts)

            preds = np.array([[0, 0, 1, 1, 4, 1]])
            
            metric_fn.add(preds, gts)

    result = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.0+stride, stride), mpolicy='soft')
    average_precision = {}
    for clsid in range(num_classes):
        average_precision[clsid] = []
    for k, v in result.items():
        if k=='mAP':
            continue
        for clsid in range(num_classes):
            average_precision[clsid].append(v[clsid]['ap'])

    output = {
        'mAP': result['mAP'],
    }
    for clsid in range(num_classes):
        average_precision[clsid] = np.mean(average_precision[clsid])
        if clsid < len(classes):
            output[classes[clsid]] = average_precision[clsid]
        elif clsid == 4:
            output['none'] = average_precision[clsid]
        else:
            output['opacity'] = average_precision[clsid]
    return output