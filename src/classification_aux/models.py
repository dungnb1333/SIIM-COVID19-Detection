from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.linknet.decoder import LinknetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init

class PretrainModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b7",
        encoder_weights: Optional[str] = None,
        classes: int = 4,
        in_features: int = 2560, 
        pretrained_path=None, 
        pretrained_num_classes=None,
    ):
        super(PretrainModel, self).__init__()
        self.in_features = in_features
        if pretrained_path is None:
            self.encoder = get_encoder(
                encoder_name,
                in_channels=3,
                depth=5,
                weights=encoder_weights,
            )
            if 'timm-efficientnet' in encoder_name:
                self.hidden_layer = nn.Sequential(*list(self.encoder.children())[-4:])
                del self.encoder.global_pool
                del self.encoder.act2
                del self.encoder.bn2
                del self.encoder.conv_head
            elif 'timm-seresnet' in encoder_name or 'timm-resnet' in encoder_name:
                self.hidden_layer = nn.AdaptiveAvgPool2d(output_size=1)
        else:
            print('Load pretrain: {}'.format(pretrained_path))
            model = PretrainModel(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=pretrained_num_classes,
                in_features=in_features, 
                pretrained_path=None, 
                pretrained_num_classes=None)
            model.load_state_dict(torch.load(pretrained_path))
            self.encoder = model.encoder
            self.hidden_layer = model.hidden_layer
            del model

        self.fc = nn.Linear(in_features, 1024, bias=True)
        self.cls_head = nn.Linear(1024, classes, bias=True)

        init.initialize_head(self.fc)
        init.initialize_head(self.cls_head)

    @autocast()
    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.hidden_layer(x)
        x = x.view(-1, self.in_features)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.cls_head(x)
        return x

class SiimCovidAuxModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b7",
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder: Optional[str] = 'unet',
        classes: int = 4,
        in_features: int = 2560,
        encoder_pretrained_path=None, 
        encoder_pretrained_num_classes=None,
        model_pretrained_path=None, 
        model_pretrained_num_classes=None,
        test_mode=False,
    ):
        super(SiimCovidAuxModel, self).__init__()
        self.in_features = in_features
        self.test_mode = test_mode

        if model_pretrained_path is None:
            if encoder_pretrained_path is None:
                model = PretrainModel(
                            encoder_name=encoder_name,
                            encoder_weights=encoder_weights,
                            classes=classes,
                            in_features=in_features, 
                            pretrained_path=None, 
                            pretrained_num_classes=None)
            else:
                print('load pretrain', encoder_pretrained_path)
                model = PretrainModel(
                            encoder_name=encoder_name,
                            encoder_weights=encoder_weights,
                            classes=encoder_pretrained_num_classes,
                            in_features=in_features, 
                            pretrained_path=None, 
                            pretrained_num_classes=None)
                model.load_state_dict(torch.load(encoder_pretrained_path))
            self.encoder = model.encoder
            self.hidden_layer = model.hidden_layer
            del model
            self.fc = nn.Linear(in_features, 1024, bias=True)
            self.cls_head = nn.Linear(1024, classes, bias=True)

            if decoder == 'unet':
                self.decoder = UnetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_batchnorm=decoder_use_batchnorm,
                    center=True if encoder_name.startswith("vgg") else False,
                    attention_type=decoder_attention_type,
                )
            elif decoder == 'unetplusplus':
                self.decoder = UnetPlusPlusDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=5,
                    use_batchnorm=decoder_use_batchnorm,
                    center=True if encoder_name.startswith("vgg") else False,
                    attention_type=decoder_attention_type,
                )
            elif decoder == 'fpn':
                self.decoder = FPNDecoder(
                    encoder_channels=self.encoder.out_channels,
                    encoder_depth=5,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="add",
                )
            elif decoder == 'linknet':
                self.decoder = LinknetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    n_blocks=5,
                    prefinal_channels=32,
                    use_batchnorm=decoder_use_batchnorm,
                )
            elif decoder == 'deeplabv3plus':
                decoder_atrous_rates = [12, 24, 36]
                encoder_output_stride = 16
                self.encoder.make_dilated(
                    stage_list=[5],
                    dilation_list=[2]
                )
                self.decoder = DeepLabV3PlusDecoder(
                    encoder_channels=self.encoder.out_channels,
                    out_channels=decoder_channels,
                    atrous_rates=decoder_atrous_rates,
                    output_stride=encoder_output_stride,
                )
            else:
                raise ValueError('Decoder error!!!')

            if decoder == 'unet' or decoder == 'unetplusplus':
                self.segmentation_head = SegmentationHead(
                    in_channels=decoder_channels[-1],
                    out_channels=1,
                    activation='sigmoid',
                    kernel_size=3,
                )
            elif decoder == 'deeplabv3plus':
                self.segmentation_head = SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    out_channels=1,
                    activation='sigmoid',
                    kernel_size=1,
                    upsampling=4,
                )
            elif decoder == 'fpn':
                self.segmentation_head = SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    out_channels=1,
                    activation='sigmoid',
                    kernel_size=1,
                    upsampling=4,
                )
            elif decoder == 'linknet':
                self.segmentation_head = SegmentationHead(
                    in_channels=32, 
                    out_channels=1, 
                    activation='sigmoid',
                    kernel_size=1
                )
            else:
                raise ValueError('Decoder error!!!')

            init.initialize_head(self.fc)
            init.initialize_head(self.cls_head)
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)
        else:
            print('Load pretrain: {}'.format(model_pretrained_path))
            model = SiimCovidAuxModel(
                encoder_name=encoder_name,
                encoder_weights=None,
                decoder=decoder,
                classes=model_pretrained_num_classes,
                in_features=in_features,
                decoder_channels=decoder_channels,
                encoder_pretrained_path=None,
                encoder_pretrained_num_classes=None,
                model_pretrained_path=None, 
                model_pretrained_num_classes=None,
                test_mode=False,
            )
            model.load_state_dict(torch.load(model_pretrained_path))

            self.encoder = model.encoder
            self.hidden_layer = model.hidden_layer
            self.decoder = model.decoder
            self.segmentation_head = model.segmentation_head
            self.fc = nn.Linear(in_features, 1024, bias=True)
            self.cls_head = nn.Linear(1024, classes, bias=True)
            del model
            init.initialize_head(self.fc)
            init.initialize_head(self.cls_head)

    @autocast()
    def forward(self, x):
        if self.test_mode:
            features = self.encoder(x)
            xcls = self.hidden_layer(features[-1])
            xcls = xcls.view(-1, self.in_features)
            xcls = self.fc(xcls)
            xcls = F.relu(xcls)
            xcls = F.dropout(xcls, p=0.5, training=self.training)
            xcls = self.cls_head(xcls)
            return xcls
        else:
            features = self.encoder(x)
            xseg = self.decoder(*features)
            xseg = self.segmentation_head(xseg)

            xcls = self.hidden_layer(features[-1])
            xcls = xcls.view(-1, self.in_features)
            xcls = self.fc(xcls)
            xcls = F.relu(xcls)
            xcls = F.dropout(xcls, p=0.5, training=self.training)
            xcls = self.cls_head(xcls)

            return xseg, xcls
