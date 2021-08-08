import numpy as np
import pandas as pd
import cv2
import os
from lxml.etree import Element, SubElement, ElementTree
from multiprocessing import Pool

class ME:
    def __init__(self, image_path, label, source, index, total):
        self.image_path = image_path
        self.label = label
        self.source = source
        self.index = index
        self.total = total

def create_ann(ele):
    print('{:06d}/{:06d}  {}'.format(ele.index, ele.total, ele.source))

    image_path = ele.image_path
    label = ele.label

    ann_dir = '/'.join(image_path.split('/')[0:-1]).replace('/images', '/labels')
    os.makedirs(ann_dir, exist_ok=True)

    yolo_ann_path = image_path.replace('/images', '/labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
    eff_ann_path = image_path.replace('/images', '/labels').replace('.png', '.xml').replace('.jpg', '.xml').replace('.jpeg', '.xml')

    arr = label.split(' ')
    nums = len(arr) // 6

    image = cv2.imread(image_path)
    height, width = image.shape[0:2]

    boxes = []
    for i in range(nums):
        class_name = arr[6*i]
        assert class_name == 'opacity'
        x1 = int(float(arr[6*i+2]))
        y1 = int(float(arr[6*i+3]))
        x2 = int(float(arr[6*i+4]))
        y2= int(float(arr[6*i+5]))

        boxes.append([x1,y1,x2,y2])

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train'
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_path.split('/')[-1]
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    
    label_file = open(yolo_ann_path, 'w')
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        xc = 0.5*(x1+x2)/width
        yc = 0.5*(y1+y2)/height
        w = (x2-x1)/width
        h = (y2-y1)/height

        label_file.write('0 {} {} {} {}\n'.format(xc, yc, w, h))

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'opacity'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(x1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(y1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(x2)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(y2)

    label_file.close()

    tree = ElementTree(node_root)
    tree.write(eff_ann_path, pretty_print=True, xml_declaration=False)
    
    return None

if __name__ == "__main__":
    os.makedirs('../../dataset/pseudo_det', exist_ok=True)
    cnt = 0
    meles = []
    for source in ['padchest', 'pneumothorax', 'vin', 'test']:
        print('*'*20, source, '*'*20)
        ext_df = pd.read_csv('../../dataset/pseudo_csv_det/{}.csv'.format(source))
        
        for _, row in ext_df.iterrows():
            if row['label'] == 'none 1 0 0 1 1':
                continue
            meles.append(ME(row['image_path'], row['label'], source, cnt, 0))
            cnt += 1
    for e in meles:
        e.total = len(meles)
    
    print(len(meles))
    p = Pool(16)
    results = p.map(func=create_ann, iterable = meles)
    p.close()

    warmup_yolo_train = open('../../dataset/pseudo_det/warmup_yolo_train.txt', 'w')
    with open('../../dataset/external_dataset/ext_csv/rsna_pneumonia_yolov5_train.txt', 'r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    for line in contents:
        warmup_yolo_train.write(line+'\n')
    with open('../../dataset/external_dataset/ext_csv/rsna_pneumonia_yolov5_valid.txt', 'r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    for line in contents:
        warmup_yolo_train.write(line+'\n')
    
    for source in ['padchest', 'pneumothorax', 'vin', 'test']:
        print('*'*20, source, '*'*20)
        ext_df = pd.read_csv('../../dataset/pseudo_csv_det/{}.csv'.format(source))
        
        eff_train = open('../../dataset/pseudo_det/{}_effdet_ext_train.txt'.format(source), 'w')
        for _, row in ext_df.iterrows():
            if row['label'] == 'none 1 0 0 1 1':
                continue
            warmup_yolo_train.write(row['image_path'] + '\n')
            if '.png' in row['image_path']:
                eff_train.write(row['image_path'].split('/')[-1].replace('.png','') + '\n')
        
        eff_train.close()
    warmup_yolo_train.close()

    warmup_yolo_valid = open('../../dataset/pseudo_det/warmup_yolo_valid.txt', 'w')
    with open('../../dataset/siim-covid19-detection/folds/yolov5_train_fold0.txt', 'r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    for line in contents:
        warmup_yolo_valid.write(line+'\n')
    with open('../../dataset/siim-covid19-detection/folds/yolov5_valid_fold0.txt', 'r') as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]
    for line in contents:
        warmup_yolo_valid.write(line+'\n')
    warmup_yolo_valid.close()