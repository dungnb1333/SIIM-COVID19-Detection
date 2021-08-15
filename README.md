# SIIM-COVID19-Detection

![Alt text](./images/header.png?raw=true "Optional Title")

Source code of the 1st place solution for [SIIM-FISABIO-RSNA COVID-19 Detection Challenge](https://www.kaggle.com/c/siim-covid19-detection/overview).

## 1.INSTALLATION
- Ubuntu 18.04.5 LTS
- CUDA 10.2
- Python 3.7.9
- python packages are detailed separately in [requirements](https://github.com/dungnb1333/SIIM-COVID19-Detection/blob/main/requirements.txt)
```
$ conda create -n envs python=3.7.9
$ conda activate envs
$ conda install -c conda-forge gdcm
$ pip install -r requirements.txt
$ pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
$ pip install git+https://github.com/bes-dev/mean_average_precision.git@930df3618c924b694292cc125114bad7c7f3097e
```

## 2.DATASET
#### 2.1 SIIM COVID 19 DATASET
- download competition dataset at [link](https://www.kaggle.com/c/siim-covid19-detection/data) then extract to ./dataset/siim-covid19-detection
```
$ cd src/prepare
$ python dicom2image_siim.py
$ python kfold_split.py
$ prepare_siim_annotation.py                        # effdet and yolo format
$ cp -r ../../dataset/siim-covid19-detection/images ../../dataset/lung_crop/.
$ python prepare_siim_lung_crop_annotation.py
```
#### 2.2 EXTERNAL DATASET
- download pneumothorax dataset at [link](https://www.kaggle.com/seesee/siim-train-test) then extract to ./dataset/external_dataset/pneumothorax/dicoms
- download pneumonia dataset at [link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) then extract to ./dataset/external_dataset/rsna-pneumonia-detection-challenge/dicoms
- download vinbigdata dataset at [link](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data) then extract to ./dataset/external_dataset/vinbigdata/dicoms
- download chest14 dataset at [link](https://nihcc.app.box.com/v/ChestXray-NIHCC) then extract to ./dataset/external_dataset/chest14/images
- download chexpert high-resolution dataset at [link](https://stanfordmlgroup.github.io/competitions/chexpert/) then extract to ./dataset/external_dataset/chexpert/train
- download padchest dataset at [link](https://bimcv.cipf.es/bimcv-projects/padchest/) then extract to ./dataset/external_dataset/padchest/images
- *<sub>Note: most of the images in bimcv and ricord duplicate with siim covid trainset and testset. To avoid data-leak when training, I didn't use them. You can use script [src/prepare/check_bimcv_ricord_dup.py](https://github.com/dungnb1333/SIIM-COVID19-Detection/blob/main/src/prepare/check_bimcv_ricord_dup.py)<sub>*
```
$ cd src/prepare
$ python dicom2image_pneumothorax.py
$ python dicom2image_pneumonia.py
$ python prepare_pneumonia_annotation.py      # effdet and yolo format
$ python dicom2image_vinbigdata.py
$ python prepare_vinbigdata.py
$ python refine_data.py                       # remove unused file in chexpert + chest14 + padchest dataset
$ python resize_padchest_pneumothorax.py
```
dataset structure should be [./dataset/dataset_structure.txt](https://github.com/dungnb1333/SIIM-COVID19-Detection/blob/main/dataset/dataset_structure.txt)
## 3.SOLUTION SUMMARY
![Alt text](./images/flowchart.png?raw=true "Optional Title")

## 4.TRAIN MODEL
### 4.1 Classification
#### 4.1.1 Multi head classification + segmentation
- Stage1
```
$ cd src/classification_aux
$ bash train_chexpert_chest14.sh              #Pretrain backbone on chexpert + chest14
$ bash train_rsnapneu.sh                      #Pretrain rsna_pneumonia
$ bash train_siim.sh                          #Train siim covid19
```
- Stage2: Generate soft-label for classification head and mask for segmentation head.\
  Output: soft-label in ./pseudo_csv/[source].csv and public test masks in ./prediction_mask/public_test/masks
```
$ bash generate_pseudo_label.sh [checkpoints_dir]
```
- Stage3: Train model on trainset + public testset, load checkpoint from previous round
```
$ bash train_pseudo.sh [previous_checkpoints_dir] [new_checkpoints_dir]
```
Rounds of pseudo labeling (stage2) and retraining (stage3) were repeated until the score on public LB didn't improve.
- For final checkpoints
```
$ bash generate_pseudo_label.sh checkpoints_v3
$ bash train_pseudo.sh checkpoints_v3 checkpoints_v4
```
- For evaluation
```
$ CUDA_VISIBLE_DEVICES=0 python evaluate.py --cfg configs/xxx.yaml --num_tta xxx
```
mAP@0.5 4 classes: negative, typical, indeterminate, atypical
|              | [SeR152-Unet](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/classification-timm-seresnet152d_320_512_unet_aux.zip) | [EB5-Deeplab](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/classification-timm-efficientnet-b5_512_deeplabv3plus_aux.zip) | [EB6-Linknet](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/classification-timm-efficientnet-b6_448_linknet_aux.zip) | [EB7-Unet++](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/classification-timm-efficientnet-b7_512_unetplusplus_aux.zip)  | Ensemble    |
| :----------- | :---------- | :---------- | :---------- | :---------- | :---------- |
| w/o TTA/8TTA | 0.575/0.584 | 0.583/0.592 | 0.580/0.587 | 0.589/0.595 | 0.595/0.598 |

*<sub>8TTA: (orig, center-crop 80%)x(None, hflip, vflip, hflip & vflip). In [final submission](https://www.kaggle.com/nguyenbadung/siim-covid19-2021), I use 4.1.2 lung detector instead of center-crop 80%<sub>*

#### 4.1.2 [Lung Detector-YoloV5](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/detection_yolov5_lung.zip)
I annotated the train data(6334 images) using [LabelImg](https://github.com/tzutalin/labelImg) and built a lung localizer. I noticed that increasing input image size improves the modeling performance and lung detector helps the model to reduce background noise.
```
$ cd src/detection_lung_yolov5
$ cd weights && bash download_coco_weights.sh && cd ..
$ bash train.sh
```
|              | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | Average |
| :----------- | :---- | :---- | :---- | :---- | :---- | :------ |
| mAP@0.5:0.95 | 0.921 | 0.931 | 0.926 | 0.923 | 0.922 | 0.9246  |
| mAP@0.5      | 0.997 | 0.998 | 0.997 | 0.996 | 0.998 | 0.9972  |

### 4.2 Opacity Detection
Rounds of pseudo labeling (stage2) and retraining (stage3) were repeated until the score on public LB didn't improve.
#### 4.2.1 YoloV5x6 768
- Stage1:
```
$ cd src/detection_yolov5
$ cd weights && bash download_coco_weights.sh && cd ..
$ bash train_rsnapneu.sh          #pretrain with rsna_pneumonia
$ bash train_siim.sh              #train with siim covid19 dataset, load rsna_pneumonia checkpoint
```
- Stage2: Generate pseudo label (boxes)
```
$ bash generate_pseudo_label.sh
```
Jump to step 4.2.4 Ensembling + Pseudo labeling
- Stage3:
```
$ bash warmup_ext_dataset.sh      #train with pseudo labeling (public-test, padchest, pneumothorax, vin) + rsna_pneumonia
$ bash train_final.sh             #train siim covid19 boxes, load warmup checkpoint
```
#### 4.2.2 EfficientDet D7 768
- Stage1:
```
$ cd src/detection_efffdet
$ bash train_rsnapneu.sh          #pretrain with rsna_pneumonia
$ bash train_siim.sh              #train with siim covid19 dataset, load rsna_pneumonia checkpoint
```
- Stage2: Generate pseudo label (boxes)
```
$ bash generate_pseudo_label.sh
```
Jump to step 4.2.4 Ensembling + Pseudo labeling
- Stage3:
```
$ bash warmup_ext_dataset.sh      #train with pseudo labeling (public-test, padchest, pneumothorax, vin) + rsna_pneumonia
$ bash train_final.sh             #train siim covid19, load warmup checkpoint
```
#### 4.2.3 FasterRCNN FPN 768 & 1024
- Stage1: train backbone of model with chexpert + chest14 -> train model with rsna pneummonia -> train model with siim, load rsna pneumonia checkpoint
```
$ cd src/detection_fasterrcnn
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/resnet200d.yaml
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/resnet101d.yaml
$ CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/resnet200d.yaml
$ CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/resnet101d.yaml
$ CUDA_VISIBLE_DEVICES=0 python train_siim.py --cfg configs/resnet200d.yaml --folds 0 1 2 3 4 --SEED 123
$ CUDA_VISIBLE_DEVICES=0 python train_siim.py --cfg configs/resnet101d.yaml --folds 0 1 2 3 4 --SEED 123
```
*<sub>Note: Change SEED if training script runs into issue related to augmentation (boundingbox area=0) and comment/uncomment the following code if training script runs into issue related to resource limit<sub>*
```python
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
```
- Stage2: Generate pseudo label (boxes)
```
$ bash generate_pseudo_label.sh
```
Jump to step 4.2.4 Ensembling + Pseudo labeling
- Stage3:
```
$ CUDA_VISIBLE_DEVICES=0 python warmup_ext_dataset.py --cfg configs/resnet200d.yaml
$ CUDA_VISIBLE_DEVICES=0 python warmup_ext_dataset.py --cfg configs/resnet101d.yaml
$ CUDA_VISIBLE_DEVICES=0 python train_final.py --cfg configs/resnet200d.yaml
$ CUDA_VISIBLE_DEVICES=0 python train_final.py --cfg configs/resnet101d.yaml
```
#### 4.2.4 Ensembling + Pseudo labeling
Keep images that meet the conditions: negative prediction < 0.3 and maximum of (typical, indeterminate, atypical) predicion > 0.7. Then choose 2 boxes with the highest confidence as pseudo labels for each image.

*<sub>Note: This step requires at least 128 GB of RAM <sub>*
```
$ cd ./src/detection_make_pseudo
$ python make_pseudo.py
$ python make_annotation.py            
```

#### 4.2.5 Detection Performance
|                  | [YoloV5x6 768](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/detection_yolov5.zip) | [EffdetD7 768](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/detection_efficientdet.zip) | [F-RCNN R200 768](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/detection_fasterrcnn_resnet200d_768.zip) | [F-RCNN R101 1024](https://github.com/dungnb1333/SIIM-COVID19-Detection/releases/download/v0.1/detection_fasterrcnn_resnet101d_1024.zip) |
| :--------------- | :----------- | :----------- | :-------------- | :--------------- |
| mAP@0.5 TTA      | 0.580        | 0.594        | 0.592           | 0.596            |

## 5.FINAL SUBMISSION
[siim-covid19-2021](https://www.kaggle.com/nguyenbadung/siim-covid19-2021?scriptVersionId=69474844) Public LB: 0.658 / Private LB: 0.635\
[demo notebook](https://github.com/dungnb1333/SIIM-COVID19-Detection/blob/main/src/demo_notebook/demo.ipynb) to visualize output of models

## 6.AWESOME RESOURCES
[Pytorch](https://github.com/pytorch/pytorch)✨\
[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)✨\
[Segmentation models](https://github.com/qubvel/segmentation_models.pytorch)✨\
[EfficientDet](https://github.com/rwightman/efficientdet-pytorch)✨\
[YoloV5](https://github.com/ultralytics/yolov5)✨\
[FasterRCNN FPN](https://github.com/pytorch/vision/tree/master/torchvision/models/detection)✨\
[Albumentations](https://github.com/albumentations-team/albumentations)✨\
[Weighted boxes fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)✨
