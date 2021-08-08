import argparse
import numpy as np
import pandas as pd
import torch
import os
import torch
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

classes = [
    'Negative for Pneumonia',
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance'
]

if __name__ == "__main__":
    os.makedirs('../../dataset/pseudo_csv_det', exist_ok=True)

    for source in ['pneumothorax', 'padchest','vin', 'test']:
        if source == 'padchest':
            ext_df = pd.read_csv('../../dataset/external_dataset/ext_csv/padchest.csv')
        elif source == 'pneumothorax':
            ext_df = pd.read_csv('../../dataset/external_dataset/ext_csv/pneumothorax.csv')
        elif source == 'vin':
            ext_df = pd.read_csv('../../dataset/external_dataset/ext_csv/vin.csv')
        elif source == 'test':
            ext_df = pd.read_csv('../../dataset/siim-covid19-detection/test_meta.csv')
        else:
            raise ValueError('source !!!')
        
        print('*'*20, source, '*'*20)
        print(ext_df.shape)

        eb5_study_pred = torch.load('../classification_aux/predictions/timm-efficientnet-b5_512_deeplabv3plus_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']
        eb6_study_pred = torch.load('../classification_aux/predictions/timm-efficientnet-b6_448_linknet_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']
        eb7_study_pred = torch.load('../classification_aux/predictions/timm-efficientnet-b7_512_unetplusplus_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']
        sr152_study_pred = torch.load('../classification_aux/predictions/timm-seresnet152d_320_512_unet_aux_fold0_1_2_3_4_{}_pred_8tta.pth'.format(source))['pred_dict']

        ed7_768_image_pred = torch.load('../detection_efffdet/predictions/tf_efficientdet_d7_768_fold0_1_2_3_4_{}_pred.pth'.format(source))
        yolov5_768_image_pred = torch.load('../detection_yolov5/predictions/yolov5x6_768_fold0_1_2_3_4_{}_pred.pth'.format(source))
        fasterrcnn_r200d_image_pred = torch.load('../detection_fasterrcnn/predictions/resnet200d_768_fold0_1_2_3_4_{}_pred.pth'.format(source))
        fasterrcnn_r101_image_pred = torch.load('../detection_fasterrcnn/predictions/resnet101d_1024_fold0_1_2_3_4_{}_pred.pth'.format(source))

        output = []
        for _, row in tqdm(ext_df.iterrows(), total=len(ext_df)):
            if source == 'test':
                image_path = '../../dataset/siim-covid19-detection/images/test/{}.png'.format(row['imageid'])
                cls_pred =  0.3*eb5_study_pred[row['imageid']] + \
                            0.2*eb6_study_pred[row['imageid']] + \
                            0.2*eb7_study_pred[row['imageid']] + \
                            0.3*sr152_study_pred[row['imageid']]
            
                if cls_pred[0] < 0.3 and np.max(cls_pred[1:]) > 0.7:
                    boxes1, scores1, labels1, img_width, img_height = ed7_768_image_pred[row['imageid']]

                    boxes2, scores2, labels2, img_width2, img_height2 = yolov5_768_image_pred[row['imageid']]
                    assert img_width2 == img_width and img_height2 == img_height
                    
                    boxes3, scores3, labels3, img_width3, img_height3 = fasterrcnn_r200d_image_pred[row['imageid']]
                    assert img_width3 == img_width and img_height3 == img_height

                    boxes4, scores4, labels4, img_width4, img_height4 = fasterrcnn_r101_image_pred[row['imageid']]
                    assert img_width4 == img_width and img_height4 == img_height

                    boxes = boxes1 + boxes2 + boxes3 + boxes4
                    labels = labels1 + labels2 + labels3 + labels4

                    scores3_tmp = []
                    for s in scores3:
                        tmp = [x*0.78 for x in s]
                        scores3_tmp.append(tmp)
                    scores3 = scores3_tmp

                    scores4_tmp = []
                    for s in scores4:
                        tmp = [x*0.78 for x in s]
                        scores4_tmp.append(tmp)
                    scores4 = scores4_tmp

                    scores = scores1 + scores2 + scores3 + scores4
                    
                    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=0.6)
                    assert np.mean(labels) == 0
                    boxes = boxes.clip(0,1)

                    boxes[:,[0,2]] = boxes[:,[0,2]]*float(img_width)
                    boxes[:,[1,3]] = boxes[:,[1,3]]*float(img_height)
                    boxes = boxes.astype(int)

                    idx = np.argsort(scores)[::-1]
                    scores = scores[idx][0:2]
                    boxes = boxes[idx][0:2]
                    if np.min(scores) < 0.2:
                        continue
                    
                    opacity_pred = []
                    for box, score in zip(boxes, scores):
                        opacity_pred.append('opacity {} {} {} {} {}'.format(score, box[0], box[1], box[2],box[3]))
                    if len(opacity_pred) == 0:
                        continue
                    opacity_pred = ' '.join(opacity_pred)
                    output.append([image_path] + list(cls_pred) + [opacity_pred])

            else:
                image_path = row['image_path']
                cls_pred =  0.3*eb5_study_pred[row['image_path']] + \
                            0.2*eb6_study_pred[row['image_path']] + \
                            0.2*eb7_study_pred[row['image_path']] + \
                            0.3*sr152_study_pred[row['image_path']]
            
                if cls_pred[0] < 0.3 and np.max(cls_pred[1:]) > 0.7:
                    boxes1, scores1, labels1, img_width, img_height = ed7_768_image_pred[row['image_path']]

                    boxes2, scores2, labels2, img_width2, img_height2 = yolov5_768_image_pred[row['image_path']]
                    assert img_width2 == img_width and img_height2 == img_height
                    
                    boxes3, scores3, labels3, img_width3, img_height3 = fasterrcnn_r200d_image_pred[row['image_path']]
                    assert img_width3 == img_width and img_height3 == img_height

                    boxes4, scores4, labels4, img_width4, img_height4 = fasterrcnn_r101_image_pred[row['image_path']]
                    assert img_width4 == img_width and img_height4 == img_height

                    boxes = boxes1 + boxes2 + boxes3 + boxes4
                    labels = labels1 + labels2 + labels3 + labels4

                    scores3_tmp = []
                    for s in scores3:
                        tmp = [x*0.78 for x in s]
                        scores3_tmp.append(tmp)
                    scores3 = scores3_tmp

                    scores4_tmp = []
                    for s in scores4:
                        tmp = [x*0.78 for x in s]
                        scores4_tmp.append(tmp)
                    scores4 = scores4_tmp

                    scores = scores1 + scores2 + scores3 + scores4
                    
                    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=0.6)
                    assert np.mean(labels) == 0
                    boxes = boxes.clip(0,1)

                    boxes[:,[0,2]] = boxes[:,[0,2]]*float(img_width)
                    boxes[:,[1,3]] = boxes[:,[1,3]]*float(img_height)
                    boxes = boxes.astype(int)

                    idx = np.argsort(scores)[::-1]
                    scores = scores[idx][0:2]
                    boxes = boxes[idx][0:2]
                    if np.min(scores) < 0.2:
                        continue
                    
                    opacity_pred = []
                    for box, score in zip(boxes, scores):
                        opacity_pred.append('opacity {} {} {} {} {}'.format(score, box[0], box[1], box[2],box[3]))
                    if len(opacity_pred) == 0:
                        continue
                    opacity_pred = ' '.join(opacity_pred)
                    output.append([image_path] + list(cls_pred) + [opacity_pred])
        output = np.array(output)
        pseudo_df = pd.DataFrame()
        pseudo_df['image_path'] = np.array(output[:,0], dtype=str)
        pseudo_df[classes] = np.array(output[:,1:5], dtype=float)
        pseudo_df['label'] = np.array(output[:,5], dtype=str)
        
        print(pseudo_df.shape, ext_df.shape)
        pseudo_df.to_csv('../../dataset/pseudo_csv_det/{}.csv'.format(source), index=False)
        