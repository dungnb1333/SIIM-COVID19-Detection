import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.parallel
from contextlib import suppress
import os

from effdet import create_model, create_loader
from effdet.data import resolve_input_config
from timm.utils import setup_default_logging
from timm.models.layers import set_layer_config

from dataset import SiimCovidDataset
from utils import seed_everything, refine_det

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument("--folds", default=[0,1,2,3,4], nargs="+", type=int)
parser.add_argument("--frac", default=1.0, type=float)
parser.add_argument('--image-size', type=int, default=None)
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d7',
                    help='model architecture (default: tf_efficientdet_d7)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
args = parser.parse_args()

SEED = 123
seed_everything(SEED)

if __name__ == "__main__":
    os.makedirs('predictions', exist_ok = True)

    setup_default_logging()

    if args.amp:
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.prefetcher = not args.no_prefetcher

    test_df = pd.read_csv('../../dataset/siim-covid19-detection/test_meta.csv')
    if args.frac != 1:
        test_df = test_df.sample(frac=args.frac).reset_index(drop=True)

    models = {}
    for fold in args.folds:
        print('*'*20, 'Fold {}'.format(fold), '*'*20)
        CHECKPOINT = 'checkpoints/{}_{}_fold{}/model_best.pth.tar'.format(args.model, args.image_size, fold)

        # create model
        with set_layer_config(scriptable=args.torchscript):
            extra_args = {}
            bench = create_model(
                args.model,
                bench_task='predict',
                image_size=args.image_size,
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                redundant_bias=args.redundant_bias,
                soft_nms=args.soft_nms,
                checkpoint_path=CHECKPOINT,
                checkpoint_ema=args.use_ema,
                **extra_args,
            )
        model_config = bench.config

        param_count = sum([m.numel() for m in bench.parameters()])
        print('Model %s created, param count: %d' % (args.model, param_count))

        bench = bench.cuda()

        amp_autocast = suppress
        if args.apex_amp:
            bench = amp.initialize(bench, opt_level='O1')
            print('Using NVIDIA APEX AMP. Validating in mixed precision.')
        elif args.native_amp:
            amp_autocast = torch.cuda.amp.autocast
            print('Using native Torch AMP. Validating in mixed precision.')
        else:
            print('AMP not enabled. Validating in float32.')
        input_config = resolve_input_config(args, model_config)
        bench.eval()

        models[fold] = bench

    dataset = SiimCovidDataset(df=test_df, images_dir='../../dataset/siim-covid19-detection/images/test', image_size=args.image_size)
        
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    predict_dict = {}
    for input, target in tqdm(loader):
        image_idxs = target['img_idx'].data.cpu().numpy().tolist()
        image_sizes = target['img_size'].data.cpu().numpy().tolist()

        with amp_autocast(), torch.no_grad():
            for fold in args.folds:
                dets = models[fold](input, img_info=target).data.cpu().numpy()
                flip_dets = models[fold](torch.flip(input, dims=(3,)).contiguous(), img_info=target).data.cpu().numpy()
                for idx, det_pred, flip_det_pred, img_size in zip(image_idxs, dets, flip_dets, image_sizes):
                    imageid = test_df.loc[idx, 'imageid']
                    if imageid not in list(predict_dict.keys()):
                        predict_dict[imageid] = [[],[],[], img_size[0], img_size[1]]
                    
                    box_pred = det_pred[:,:4].astype(float)
                    box_pred[:,[0,2]] = box_pred[:,[0,2]]/float(img_size[0])
                    box_pred[:,[1,3]] = box_pred[:,[1,3]]/float(img_size[1])
                    score_pred = det_pred[:,4]
                    label_pred = det_pred[:,5].astype(int) - 1
                    box_pred, label_pred, score_pred = refine_det(box_pred, label_pred, score_pred)

                    flip_box_pred = flip_det_pred[:,:4].astype(float)
                    flip_box_pred[:,[0,2]] = flip_box_pred[:,[0,2]]/float(img_size[0])
                    flip_box_pred[:,[1,3]] = flip_box_pred[:,[1,3]]/float(img_size[1])
                    flip_box_pred[:,[0,2]] = 1 - flip_box_pred[:,[0,2]]
                    flip_score_pred = flip_det_pred[:,4]
                    flip_label_pred = flip_det_pred[:,5].astype(int) - 1
                    flip_box_pred, flip_label_pred, flip_score_pred = refine_det(flip_box_pred, flip_label_pred, flip_score_pred)

                    predict_dict[imageid][0] += [box_pred, flip_box_pred]
                    predict_dict[imageid][1] += [score_pred, flip_score_pred]
                    predict_dict[imageid][2] += [label_pred, flip_label_pred]
    
    pred_dict_path = 'predictions/{}_{}_fold{}_test_pred.pth'.format(args.model, args.image_size, '_'.join(str(x) for x in args.folds))
    torch.save(predict_dict, pred_dict_path)
