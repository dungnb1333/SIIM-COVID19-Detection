CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/resnet101d.yaml
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/resnet200d.yaml
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/resnet101d.yaml
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/resnet200d.yaml
