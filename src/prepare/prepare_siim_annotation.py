import pandas as pd
import cv2
import os
from lxml.etree import Element, SubElement, ElementTree
from multiprocessing import Pool

def create_ann(ele):
    ann_path = '../../dataset/siim-covid19-detection/labels/train/{}.xml'.format(ele.imageid)
    label_path = '../../dataset/siim-covid19-detection/labels/train/{}.txt'.format(ele.imageid)

    image = cv2.imread(ele.image_path)
    height, width = image.shape[0:2]

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train'
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = '{}.jpg'.format(ele.imageid)
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    
    label_file = open(label_path, 'w')
    for box in ele.boxes:
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

        cv2.rectangle(image,  (x1, y1), (x2, y2), (255,0,0), 5)
    label_file.close()

    tree = ElementTree(node_root)
    tree.write(ann_path, pretty_print=True, xml_declaration=False)
    
    return None

class ME:
    def __init__(self, image_path, imageid, boxes):
        self.image_path = image_path
        self.imageid = imageid
        self.boxes = boxes

if __name__ == '__main__':
    os.makedirs('../../dataset/siim-covid19-detection/labels/train', exist_ok=True)
    os.makedirs('../../dataset/siim-covid19-detection/folds', exist_ok=True)

    df = pd.read_csv('../../dataset/siim-covid19-detection/train_kfold.csv')
    df = df.loc[df['hasbox'] == True].reset_index(drop=True)
    for fold in range(5):
        tmp_df = df.loc[df['fold'] == fold]

        meles = []
        for _, row in tmp_df.iterrows():
            image_path = '../../dataset/siim-covid19-detection/images/train/{}.png'.format(row['imageid'])

            arr = row['label'].split(' ')
            nums = len(arr) // 6
            assert nums > 0

            boxes = []
            for i in range(nums):
                class_name = arr[6*i]
                conf = int(arr[6*i+1])
                assert class_name == 'opacity'
                x1 = float(arr[6*i+2])
                y1 = float(arr[6*i+3])
                x2 = float(arr[6*i+4])
                y2= float(arr[6*i+5])
                if x1 >= x2 or y1 >= y2:
                    continue
                boxes.append([x1, y1, x2, y2])
            if len(boxes) > 0:
                meles.append(ME(image_path, row['imageid'], boxes))
        
        p = Pool(16)
        results = p.map(func=create_ann, iterable = meles)
        p.close()

    for fold in range(5):
        val_df = df.loc[df['fold'] == fold].sample(frac=1).reset_index(drop=True)
        train_df = df.loc[df['fold'] != fold].sample(frac=1).reset_index(drop=True)
        
        yv5_tf = open("../../dataset/siim-covid19-detection/folds/yolov5_train_fold{}.txt".format(fold), "w")
        effdet_tf = open("../../dataset/siim-covid19-detection/folds/effdet_train_fold{}.txt".format(fold), "w")
        for _, row in train_df.iterrows():
            image_path = '../../dataset/siim-covid19-detection/images/train/{}.png'.format(row['imageid'])
            yv5_tf.write(image_path + '\n')
            effdet_tf.write(row['imageid'] + '\n')
        yv5_tf.close()
        effdet_tf.close()

        yv5_vf = open("../../dataset/siim-covid19-detection/folds/yolov5_valid_fold{}.txt".format(fold), "w")
        effdet_vf = open("../../dataset/siim-covid19-detection/folds/effdet_valid_fold{}.txt".format(fold), "w")
        for _, row in val_df.iterrows():
            image_path = '../../dataset/siim-covid19-detection/images/train/{}.png'.format(row['imageid'])
            yv5_vf.write(image_path + '\n')
            effdet_vf.write(row['imageid'] + '\n')
        yv5_vf.close()
        effdet_vf.close()