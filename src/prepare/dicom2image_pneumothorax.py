import numpy as np 
import os 
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
from multiprocessing import Pool

def dicom2img(dicom_path):
    dcm_file = pydicom.read_file(dicom_path)
    data = apply_voi_lut(dcm_file.pixel_array, dcm_file)

    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    SOPInstanceUID = dcm_file.SOPInstanceUID

    image_path = '../../dataset/external_dataset/pneumothorax/images/{}.png'.format(SOPInstanceUID)
    cv2.imwrite(image_path, data)
    return image_path
    
if __name__ == '__main__':
    os.makedirs('../../dataset/external_dataset/pneumothorax/images', exist_ok=True)

    dcm_paths = []
    for rdir, _, files in os.walk('../../dataset/external_dataset/pneumothorax/dicoms/dicom-images-train'):
        for file in files:
            if '.dcm' in file:
                dcm_path = os.path.join(rdir, file)
                dcm_paths.append(dcm_path)
    for rdir, _, files in os.walk('../../dataset/external_dataset/pneumothorax/dicoms/dicom-images-test'):
        for file in files:
            if '.dcm' in file:
                dcm_path = os.path.join(rdir, file)
                dcm_paths.append(dcm_path)
    print(len(dcm_paths))
    p = Pool(16)
    results = p.map(func=dicom2img, iterable = dcm_paths)
    p.close()

    
