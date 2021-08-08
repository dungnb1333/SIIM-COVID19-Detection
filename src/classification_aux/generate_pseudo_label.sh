#predict softlabel for public test
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/eb5_512_deeplabv3plus.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/eb6_448_linknet.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/eb7_512_unetplusplus.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_test.py --cfg configs/seresnet152d_512_unet.yaml --ckpt_dir $1
#predict mask for public test using segmentation head
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/eb5_512_deeplabv3plus.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/eb6_448_linknet.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/eb7_512_unetplusplus.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_test_seg.py --cfg configs/seresnet152d_512_unet.yaml --ckpt_dir $1
#ensemble 4 models eb5, eb6, eb7, seresnet152
python ensemble_pseudo_test.py
#predict softlabel for padchest, pneumothorax, vin
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/eb5_512_deeplabv3plus.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/eb6_448_linknet.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/eb7_512_unetplusplus.yaml --ckpt_dir $1
CUDA_VISIBLE_DEVICES=0 python predict_ext.py --cfg configs/seresnet152d_512_unet.yaml --ckpt_dir $1
python ensemble_pseudo_ext.py