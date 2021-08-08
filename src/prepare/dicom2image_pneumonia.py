import pandas as pd
import os
import numpy as np
import pydicom
import cv2
from multiprocessing import Pool
from pydicom.pixel_data_handlers.util import apply_voi_lut

class ME:
    def __init__(self, patientid, file_path, type):
        self.patientid = patientid
        self.file_path = file_path
        self.type = type

def dicom2image(ele):
    dcm_file = pydicom.read_file(ele.file_path)
    Modality = dcm_file.Modality
    PatientSex = dcm_file.PatientSex
    BodyPartExamined = dcm_file.BodyPartExamined
    ImagerPixelSpacingX, ImagerPixelSpacingY = float(dcm_file.PixelSpacing[0]), float(dcm_file.PixelSpacing[1])
    PatientID = dcm_file.PatientID
    StudyInstanceUID = dcm_file.StudyInstanceUID
    SOPInstanceUID = dcm_file.SOPInstanceUID
    assert ele.patientid == PatientID
    
    data = apply_voi_lut(dcm_file.pixel_array, dcm_file)

    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    image_path = '../../dataset/external_dataset/rsna-pneumonia-detection-challenge/images/{}/{}.png'.format(ele.type, PatientID)
    cv2.imwrite(image_path, data)

    return [StudyInstanceUID, PatientID, SOPInstanceUID, Modality, PatientSex, BodyPartExamined, ImagerPixelSpacingX, ImagerPixelSpacingY]

if __name__ == '__main__':
    os.makedirs('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/images/train', exist_ok=True)
    os.makedirs('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/images/test', exist_ok=True)

    ################ TEST ################
    meles = []
    df = pd.read_csv('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/dicoms/stage_2_sample_submission.csv')
    for patientId in np.unique(df.patientId.values):
        dcm_path = '../../dataset/external_dataset/rsna-pneumonia-detection-challenge/dicoms/stage_2_test_images/{}.dcm'.format(patientId)
        meles.append(ME(patientId, dcm_path, 'test'))
    
    p = Pool(16)
    results = p.map(func=dicom2image, iterable = meles)
    p.close()
    test_df = pd.DataFrame(
        data=np.array(results), 
        columns=['studyid', 'patientid', 'imageid', 'Modality', 'PatientSex', 'BodyPartExamined', 'ImagerPixelSpacingX', 'ImagerPixelSpacingY'])
    test_df.to_csv('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/test_meta.csv', index=False)

    ################ TRAIN ################
    meles = []
    df = pd.read_csv('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/dicoms/stage_2_train_labels.csv')
    for patientId in np.unique(df.patientId.values):
        dcm_path = '../../dataset/external_dataset/rsna-pneumonia-detection-challenge/dicoms/stage_2_train_images/{}.dcm'.format(patientId)
        meles.append(ME(patientId, dcm_path, 'train'))

    p = Pool(16)
    results = p.map(func=dicom2image, iterable = meles)
    p.close()
    train_df = pd.DataFrame(
        data=np.array(results), 
        columns=['studyid', 'patientid', 'imageid', 'Modality', 'PatientSex', 'BodyPartExamined', 'ImagerPixelSpacingX', 'ImagerPixelSpacingY'])
    train_df.to_csv('../../dataset/external_dataset/rsna-pneumonia-detection-challenge/train_meta.csv', index=False)