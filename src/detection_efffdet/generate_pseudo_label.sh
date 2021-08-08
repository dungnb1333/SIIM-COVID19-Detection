CUDA_VISIBLE_DEVICES=0 python predict_test.py --model tf_efficientdet_d7 --amp --use-ema --num-classes 1 --native-amp -b 16 --folds 0 1 2 3 4 --image-size 768
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --model tf_efficientdet_d7 --amp --use-ema --num-classes 1 --native-amp -b 16 --folds 0 1 2 3 4 --image-size 768
