import pandas as pd
import cv2
import os
from lxml.etree import Element, SubElement, ElementTree
from multiprocessing import Pool


def create_ann(ele):
    ann_path = '../../dataset/external_dataset/rsna-pneumonia-detection-challenge/labels/train/{}.xml'.format(ele.patientId)
    label_path = '../../dataset/external_dataset/rsna-pneumonia-detection-challenge/labels/train/{}.txt'.format(ele.patientId)

    image = cv2.imread(ele.image_path)
    height, width = image.shape[0:2]

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'train'
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = '{}.jpg'.format(ele.patientId)
    
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
    
    return ele.patientId

class ME:
    def __init__(self, image_path, patientId, boxes):
        self.image_path = image_path
        self.patientId = patientId
        self.boxes = boxes
        

if __name__ == '__main__':
    os.makedirs('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/labels/train', exist_ok=True)

    df = pd.read_csv('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/dicoms/stage_2_train_labels.csv')
    df = df.loc[df['Target'] == 1].reset_index(drop=True)

    meles = []
    for patientId, grp in df.groupby('patientId'):
        image_path = '../../dataset/external_dataset/rsna-pneumonia-detection-challenge/images/train/{}.png'.format(patientId)
        boxes = []
        for _, row in grp.iterrows():
            x,y,width,height = float(row['x']), float(row['y']), float(row['width']), float(row['height'])
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            if x1 >= x2 or y1 >= y2:
                continue
            boxes.append([x1, y1, x2, y2])
        if len(boxes) > 0:
            meles.append(ME(image_path, patientId, boxes))

    p = Pool(16)
    results = p.map(func=create_ann, iterable = meles)
    p.close()