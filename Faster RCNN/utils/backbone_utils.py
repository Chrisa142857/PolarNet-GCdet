#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import densenet

from .layergetter import IntermediateLayerGetter, DenseNetLayerGetter
from .fpn import FeaturePyramidNetwork, MaxpoolOnP5, Bottom_up_path
from .fpn import AttFeaturePyramidNetwork
from .fpn import PANModule
from .misc import FrozenBatchNorm2d


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, return_layers,
                 in_channels_list, out_channel):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        # self.afpn = AttFeaturePyramidNetwork(in_channels_list,
        #                                      out_channel,
                                             # extra_block=MaxpoolOnP5())
        # self.pan = PANModule(in_channels_list,
        #                      out_channel)
        # self.bottom_up = Bottom_up_path([256,256,256,256],
        #                                 out_channel,
        #                                 extra_block=MaxpoolOnP5())
        
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        # x = self.afpn(x)
        x = self.fpn(x)
        # x = self.bottom_up(x)
        return x

class BackboneWithAFPN(nn.Module):

    def __init__(self, backbone, return_layers,
                 in_channels_list, out_channel):
        super(BackboneWithAFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers)
        # self.fpn = FeaturePyramidNetwork(in_channels_list,
        #                                  out_channel,
        #                                  extra_block=MaxpoolOnP5())
        self.afpn = AttFeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=MaxpoolOnP5())
        # self.pan = PANModule(in_channels_list,
        #                      out_channel)
        # self.bottom_up = Bottom_up_path([256,256,256,256],
        #                                 out_channel,
        #                                 extra_block=MaxpoolOnP5())
        
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.afpn(x)
        # x = self.fpn(x)
        # x = self.bottom_up(x)
        return x


class BackboneWithFPNForDensenet(nn.Module):

    def __init__(self, backbone, in_channels_list, out_channel):
        super(BackboneWithFPNForDensenet, self).__init__()
        self.body = DenseNetLayerGetter(backbone)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        self.afpn = AttFeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=MaxpoolOnP5())
        # self.pan = PANModule(in_channels_list,
        #                      out_channel)
        # self.bottom_up = Bottom_up_path([256,256,256,256],
        #                                 out_channel,
        #                                 extra_block=MaxpoolOnP5())
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.afpn(x)
        # x = self.fpn(x)
        # x = self.bottom_up(x)
        return x


# def resnet_fpn_backbone(backbone_name, pretrained,
#                         norm_layer=FrozenBatchNorm2d):

#     backbone = resnet.__dict__[backbone_name](
#         pretrained=pretrained,
#         norm_layer=norm_layer
#     )
#     for name, param in backbone.named_parameters():
#         if "layer2" not in name and "layer3" not in name and "layer4" not in name:
#             param.requires_grad_(False)

#     return_layers = {"layer1": "0", "layer2": "1",
#                      "layer3": "2", "layer4": "3"}

#     in_channels_stage2 = backbone.inplanes // 8
#     in_channels_list = [
#         in_channels_stage2,
#         in_channels_stage2 * 2,
#         in_channels_stage2 * 4,
#         in_channels_stage2 * 8,
#     ]
#     out_channel = 256
#     return BackboneWithFPN(backbone, return_layers,
#                           in_channels_list, out_channel)
def resnet_fpn_backbone(backbone_name, pretrained, use_attn=0):

    backbone = resnet.__dict__[backbone_name](
        pretrained=True
    )
    for name, param in backbone.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name:
            param.requires_grad_(False)

    return_layers = {"layer1": "0", "layer2": "1",
                     "layer3": "2", "layer4": "3"}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channel = 256

    if use_attn == 1:
        return BackboneWithAFPN(backbone, return_layers,
                          in_channels_list, out_channel)
    else:
        return BackboneWithFPN(backbone, return_layers,
                          in_channels_list, out_channel)


def densenet_fpn_backbone(backbone_name, pretrained):
    backbone = densenet.__dict__[backbone_name](
        pretrained=pretrained
    )
    for name, param in backbone.features.named_parameters():
        if "denseblock" not in name and "transition" not in name:
            param.requires_grad_(False)


    # in_channels_list = [128, 256, 896, 1920]
    in_channels_list = {
        'densenet121': [128, 256, 512, 1024],
        'densenet161': [192, 384, 1056, 2208],
        'densenet169': [128, 256, 896, 1920],
    }
    in_channels_list = in_channels_list[backbone_name]
    out_channel = 256
    return BackboneWithFPNForDensenet(backbone,
                                      in_channels_list,
                                      out_channel)


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 3, 224, 224)
    net = resnet_fpn_backbone("resnet50", True)
    # net = densenet_fpn_backbone("densenet161", True)
    out = net(x)
    import ipdb;ipdb.set_trace()

