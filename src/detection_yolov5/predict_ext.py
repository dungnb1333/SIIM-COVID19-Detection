import argparse
import os 
from tqdm import tqdm
import torch
import pandas as pd 
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

parser = argparse.ArgumentParser()
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--sources", default=['padchest', 'pneumothorax', 'vin'], nargs="+", type=str)
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument("--workers", default=4, type=int)
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
parser.add_argument('--frac', default=1.0, type=float)
opt = parser.parse_args()
print(opt)

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

if __name__ == "__main__":
    os.makedirs('predictions', exist_ok = True)

    device = select_device(opt.device)
    stride = None
    models = {}
    for fold in opt.folds:
        print('*'*20, 'Fold {}'.format(fold), '*'*20)
        CHECKPOINT = 'runs/train/fold{}/weights/best.pt'.format(fold)

        model = attempt_load(CHECKPOINT, map_location=device)  # load FP32 model
        if stride is None:
            stride = int(model.stride.max())  # model stride
        else:
            assert stride == int(model.stride.max())
        
        model.half()  # to FP16
        model.eval()
        models[fold] = model

    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    for source in opt.sources:
        print('*'*20, source, '*'*20)
        if source == 'padchest':
            img_source = '../../dataset/external_dataset/padchest/images'
        elif source == 'pneumothorax':
            img_source = '../../dataset/external_dataset/pneumothorax/images'
        elif source == 'vin':
            img_source = '../../dataset/external_dataset/vinbigdata/images'
        else:
            raise ValueError('source !!!')

        predict_dict = {}
        dataset = LoadImages(img_source, img_size=imgsz, stride=stride)

        for path, img, im0s, vid_cap in tqdm(dataset):
            path = '../../dataset'+ path.split('../../dataset')[-1]

            img_height, img_width = im0s.shape[0:2]

            if path not in list(predict_dict.keys()):
                predict_dict[path] = [[],[],[], img_width, img_height]

            img = torch.from_numpy(img).to(device)
            img = img.half()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            for fold in opt.folds:
                with torch.no_grad():
                    pred = models[fold](img, augment=True)[0]
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=opt.agnostic_nms)

                boxes = []
                scores = []
                labels = []
                for det in pred:
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                        det = det.data.cpu().numpy()

                        box_pred = det[:,:4].astype(float)
                        box_pred[:,[0,2]] = box_pred[:,[0,2]]/float(img_width)
                        box_pred[:,[1,3]] = box_pred[:,[1,3]]/float(img_height)
                        score_pred = det[:,4]
                        label_pred = det[:,5].astype(int)
                        box_pred, label_pred, score_pred = refine_det(box_pred, label_pred, score_pred)

                        predict_dict[path][0] += [box_pred]
                        predict_dict[path][1] += [score_pred]
                        predict_dict[path][2] += [label_pred]
        
        pred_dict_path = 'predictions/yolov5x6_{}_fold{}_{}_pred.pth'.format(opt.img_size, '_'.join(str(x) for x in opt.folds), source)
        torch.save(predict_dict, pred_dict_path)
