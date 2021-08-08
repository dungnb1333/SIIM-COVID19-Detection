import pandas as pd
import os
import numpy as np
import pydicom
import cv2
from multiprocessing import Pool
from pydicom.pixel_data_handlers.util import apply_voi_lut

class ME:
    def __init__(self, StudyInstanceUID, file_path, type):
        self.StudyInstanceUID = StudyInstanceUID
        self.file_path = file_path
        self.type = type

def dicom2image(ele):
    image_id = ele.file_path.split('/')[-1].split('.')[0]
    dcm_file = pydicom.read_file(ele.file_path)
    
    PatientID = dcm_file.PatientID
    assert image_id == dcm_file.SOPInstanceUID
    series_id = dcm_file.SeriesInstanceUID

    assert ele.StudyInstanceUID == dcm_file.StudyInstanceUID

    data = apply_voi_lut(dcm_file.pixel_array, dcm_file)

    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    image_path = '../../dataset/siim-covid19-detection/images/{}/{}.png'.format(ele.type, image_id)
    cv2.imwrite(image_path, data)

    return [PatientID, ele.StudyInstanceUID, series_id, image_id, dcm_file.SeriesNumber, dcm_file.InstanceNumber]

if __name__ == '__main__':
    os.makedirs('../../dataset/siim-covid19-detection/images/train', exist_ok=True)
    os.makedirs('../../dataset/siim-covid19-detection/images/test', exist_ok=True)

    ################ TEST ################
    meles = []
    df = pd.read_csv('../../dataset/siim-covid19-detection/sample_submission.csv')
    for id in np.unique(df.id.values):
        if '_study' not in id:
            continue
        StudyInstanceUID = id.replace('_study', '')
        for rdir, _, files in os.walk('../../dataset/siim-covid19-detection/test/{}'.format(StudyInstanceUID)):
            for file in files:
                file_path = os.path.join(rdir, file)
                filename, file_extension = os.path.splitext(file_path)
                if file_extension in ['.dcm', '.dicom']:
                    meles.append(ME(StudyInstanceUID, file_path, 'test'))
    p = Pool(16)
    results = p.map(func=dicom2image, iterable = meles)
    p.close()
    test_df = pd.DataFrame(
        data=np.array(results), 
        columns=['patientid', 'studyid', 'series_id', 'imageid', 'SeriesNumber', 'InstanceNumber'])
    test_df.to_csv('../../dataset/siim-covid19-detection/test_meta.csv', index=False)

    ################ TRAIN ################
    meles = []
    df = pd.read_csv('../../dataset/siim-covid19-detection/train_study_level.csv')
    for id in np.unique(df.id.values):
        StudyInstanceUID = id.replace('_study', '')
        for rdir, _, files in os.walk('../../dataset/siim-covid19-detection/train/{}'.format(StudyInstanceUID)):
            for file in files:
                file_path = os.path.join(rdir, file)
                filename, file_extension = os.path.splitext(file_path)
                if file_extension in ['.dcm', '.dicom']:
                    meles.append(ME(StudyInstanceUID, file_path, 'train'))
    p = Pool(16)
    results = p.map(func=dicom2image, iterable = meles)
    p.close()
    train_df = pd.DataFrame(
        data=np.array(results), 
        columns=['patientid', 'studyid', 'series_id', 'imageid', 'SeriesNumber', 'InstanceNumber'])
    train_df.to_csv('../../dataset/siim-covid19-detection/train_meta.csv', index=False)