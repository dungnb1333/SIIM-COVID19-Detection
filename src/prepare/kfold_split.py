import pandas as pd 
import numpy as np
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import Counter

study_classes = [
    'Negative for Pneumonia',
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance'
]

if __name__ == '__main__':
    study_df = pd.read_csv('../../dataset/siim-covid19-detection/train_study_level.csv')
    study_df['studyid'] = study_df.apply(lambda row: row.id.split('_')[0], axis=1)
    study_df = study_df.drop('id', 1)

    image_df = pd.read_csv('../../dataset/siim-covid19-detection/train_image_level.csv')
    image_df['id'] = image_df.apply(lambda row: row.id.split('_')[0], axis=1)

    train_meta_df = pd.read_csv('../../dataset/siim-covid19-detection/train_meta.csv')
    meta_columns = train_meta_df.columns.values.tolist()

    x = []
    y = []
    for patientid, grp in tqdm(train_meta_df.groupby('patientid')):
        x.append(patientid)
        c = [0]*(len(study_classes) + 1) ###add has box
        for studyid in np.unique(grp.studyid.values):
            study_tmp_df = study_df.loc[study_df['studyid'] == studyid]
            assert len(study_tmp_df) == 1
            study_value = study_tmp_df[study_classes].values[0,:]
            for i in range(len(study_classes)):
                if study_value[i] == 1:
                    c[i] = 1
            
            image_tmp_df = image_df.loc[image_df['StudyInstanceUID'] == studyid]
            for _, row in image_tmp_df.iterrows():
                if row['label'] != 'none 1 0 0 1 1':
                    c[-1] = 1
        
        y.append(c)   
    x = np.array(x)
    y = np.array(y)

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    outputs = []
    for fold, (train_index, val_index) in enumerate(mskf.split(x, y)):
        val_df = train_meta_df.loc[train_meta_df.patientid.isin(x[val_index])]
        print('Fold {}: Patient {} | Study {}'.format(fold, len(np.unique(val_df.patientid.values)), len(np.unique(val_df.studyid.values))))
        
        for _, row in val_df.iterrows():
            meta_value = row[meta_columns].values.tolist()
            study_tmp_df = study_df.loc[study_df['studyid'] == row['studyid']]
            assert len(study_tmp_df) == 1
            study_value = list(np.squeeze(study_tmp_df[study_classes].values))
            
            image_tmp_df = image_df.loc[image_df['id'] == row['imageid']]
            assert len(image_tmp_df) == 1

            image_label = image_tmp_df.label.values[0]
            if image_label != 'none 1 0 0 1 1':
                hasbox = True
            else:
                hasbox = False
            outputs.append(meta_value+study_value+[image_label, hasbox, fold])

    kfold_df = pd.DataFrame(data=np.array(outputs), columns=[meta_columns+study_classes+['label', 'hasbox', 'fold']])
    kfold_df.to_csv('../../dataset/siim-covid19-detection/train_kfold.csv', index=False)

