CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/eb5_512_deeplabv3plus.yaml
CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/eb6_448_linknet.yaml
CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/eb7_512_unetplusplus.yaml
CUDA_VISIBLE_DEVICES=0 python train_rsnapneu.py --cfg configs/seresnet152d_512_unet.yaml