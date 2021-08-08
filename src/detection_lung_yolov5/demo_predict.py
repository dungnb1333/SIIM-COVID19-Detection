import argparse
import os 
from tqdm import tqdm
import torch
import gc
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default='', type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--output_dir", default='./det_predictions', type=str)
parser.add_argument("--output_file_name", default='yolov5_lung_test_pred.pth', type=str)
parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument("--workers", default=16, type=int)
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
opt = parser.parse_args()
# print(opt)

if __name__ == "__main__":
    os.makedirs(opt.output_dir, exist_ok = True)

    device = select_device(opt.device)
    stride = None
    CHECKPOINT = '{}/fold{}/weights/best.pt'.format(opt.ckpt_dir, opt.fold)

    model = attempt_load(CHECKPOINT, map_location=device)  # load FP32 model
    if stride is None:
        stride = int(model.stride.max())  # model stride
    else:
        assert stride == int(model.stride.max())
    
    model.half()  # to FP16
    model.eval()

    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)

    predict_dict = {}
    for path, img, im0s, vid_cap in tqdm(dataset):
        img_height, img_width = im0s.shape[0:2]

        imageid = path.split('/')[-1].replace('.png','')

        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=opt.agnostic_nms)

        assert len(pred) == 1
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        det = det.data.cpu().numpy()
        box_pred = det[:,:4].astype(int)
        score_pred = det[:,4]

        if len(score_pred) == 0:
            x1, y1, x2, y2 = 0, 0, img_width, img_height
        else:
            box_pred = box_pred[np.argmax(score_pred)]
            
            x1, y1, x2, y2 = box_pred

            x1 = min(max(0, x1), img_width)
            x2 = min(max(0, x2), img_width)
            y1 = min(max(0, y1), img_height)
            y2 = min(max(0, y2), img_height)

        predict_dict[imageid] = [x1, y1, x2, y2]
    pred_dict_path = '{}/{}'.format(opt.output_dir, opt.output_file_name)
    torch.save(predict_dict, pred_dict_path)

    del model
    del dataset
    del predict_dict
    torch.cuda.empty_cache()
    gc.collect()