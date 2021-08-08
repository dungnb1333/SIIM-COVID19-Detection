CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/eb5_512_deeplabv3plus.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/eb6_448_linknet.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/eb7_512_unetplusplus.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_chexpert_chest14.py --steps 0 1 --cfg configs/seresnet152d_512_unet.yaml