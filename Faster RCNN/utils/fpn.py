#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.jit.annotations import Dict, List


from collections import OrderedDict

from .attention_layer import CSABlock


class AttFeaturePyramidNetwork(nn.Module):
    """
    Attention feature pyramid network
    params:
        in_channels_list: list, [C2,C3,C4,C5] channels
        out_channel: num channels after this module
    """

    def __init__(self, in_channels_list, out_channel, extra_block=None):
        super(AttFeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()  # save 1x1 conv
        self.csablocks = nn.ModuleList()
        for in_channel in in_channels_list:
            inner_block = nn.Conv2d(in_channel, out_channel, 1)
            inner_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            self.inner_blocks.append(
                nn.Sequential(inner_block, inner_block_gn)
            )
            
            self.csablocks.append(CSABlock(out_channel))

        self.extra_block = extra_block


    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int) """
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_csa_blocks(self, x, y, idx):
        # type: (Tensor, int)
        num_blocks = 0
        for m in self.csablocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.csablocks:
            if i == idx:
                out = module(x, y)
            i += 1
        return out

    def forward(self, x):
        """
        forward propagate
        """
        names = list(x.keys())
        x = list(x.values())
        results = []
        # 先把最顶层的C5计算出来，因为它没有+的过程
        last_out = self.get_result_from_inner_blocks(x[-1], -1)
        results.append(last_out)
        for idx in range(len(x)-2, -1, -1):
            inner = self.get_result_from_inner_blocks(x[idx], idx)
            last_out = self.get_result_from_csa_blocks(inner, last_out, idx)
            results.insert(0, last_out)

        if self.extra_block is not None:
            names, results = self.extra_block(results, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, in_channels_list, out_channel, extra_block=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()  # 存储1x1conv
        self.layer_blocks = nn.ModuleList()  # 存储3x3conv
        for in_channel in in_channels_list:
            inner_block = nn.Conv2d(in_channel, out_channel, 1)
            # inner_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            layer_block = nn.Conv2d(out_channel, out_channel, 3, padding=1)
            # layer_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            # self.inner_blocks.append(
            #     nn.Sequential(inner_block, inner_block_gn)
            # )
            # self.layer_blocks.append(
            #     nn.Sequential(layer_block, layer_block_gn)
            # )
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_block = extra_block

    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int) """
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        # type: (Tensor, int)
        num_blocks = 0
        for m in self.layer_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        # type: (Dict[str, Tensor])
        names = list(x.keys())
        x = list(x.values())
        result = []
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        last_layer = self.get_result_from_layer_blocks(last_inner, -1)
        result.append(last_layer)
        for idx in range(len(x)-2, -1, -1):
            inner = self.get_result_from_inner_blocks(x[idx], idx)
            upsample = F.interpolate(last_inner, inner.shape[-2:])
            last_inner = inner + upsample
            layer = self.get_result_from_layer_blocks(last_inner, idx)
            result.insert(0, layer)

        if self.extra_block is not None:
            names, result = self.extra_block(result, names)

        out = OrderedDict([(k, v) for k, v in zip(names, result)])
        return out


class MaxpoolOnP5(nn.Module):
    
    def forward(self, result, name):
        # type: (List[str], List[Tensor])
        name.append("pool")
        p6 = F.max_pool2d(result[-1], 1, 2, 0)
        result.append(p6)
        return name, result


class Bottom_up_path(nn.Module):
    
    def __init__(self, in_channels_list, out_channel, extra_block=None):
        super(Bottom_up_path, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for i in range(len(in_channels_list)-1):
            inner_block = nn.Conv2d(in_channels_list[0], out_channel, 3, 2, 1)
            inner_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            layer_block = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
            layer_block_gn = nn.GroupNorm(32, out_channel, 1e-5)
            relu = nn.ReLU(inplace=True)
            self.inner_blocks.append(nn.Sequential(inner_block,
                                                   inner_block_gn,
                                                   relu))
            self.layer_blocks.append(nn.Sequential(layer_block,
                                                   layer_block_gn,
                                                   relu))

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        self.extra_block = extra_block

    def get_result_from_inner_blocks(self, x, idx):
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        num_blocks = 0
        for m in self.layer_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        result = []
        N2 = x[0]
        result.append(N2)
        pre_N3 = x[1] + self.get_result_from_inner_blocks(N2, 0)
        N3 = self.get_result_from_layer_blocks(pre_N3, 0)
        result.append(N3)
        pre_N4 = x[2] + self.get_result_from_inner_blocks(pre_N3, 1)
        N4 = self.get_result_from_layer_blocks(pre_N4, 1)
        result.append(N4)
        pre_N5 = x[3] + self.get_result_from_inner_blocks(pre_N4, 2)
        N5 = self.get_result_from_layer_blocks(pre_N5, 2)
        result.append(N5)

        if self.extra_block is not None:
            names, result = self.extra_block(result, names)

        out = OrderedDict([(k, v) for k, v in zip(names, result)])
        return out


class ConvGnRelu(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 stride=1, padding=0):
        super(ConvGnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.GroupNorm(32, out_channel),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        return out


class FPAModule(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(FPAModule, self).__init__()
        mid_channel = int(in_channel / 4)
        # conv7x7
        self.conv1 = ConvGnRelu(mid_channel, mid_channel, 7, 1, 3)
        # conv5x5
        self.conv2 = ConvGnRelu(mid_channel, mid_channel, 5, 1, 2)
        # conv1x1 to change channels
        self.conv3 = ConvGnRelu(mid_channel, out_channel, 1, 1, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGnRelu(in_channel, out_channel, 1)
        )
        self.main_branch = ConvGnRelu(in_channel, out_channel, 1)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvGnRelu(in_channel, mid_channel, 7, 1, 3)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvGnRelu(mid_channel, mid_channel, 5, 1, 2)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvGnRelu(mid_channel, mid_channel, 3, 1, 1),
            ConvGnRelu(mid_channel, mid_channel, 3, 1, 1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        # global branch
        b1 = self.global_branch(x)
        b1 = F.interpolate(b1, (h, w),
                           mode="bilinear", align_corners=True)
        # main path branch
        main_branch_out = self.main_branch(x)
        # branch1 7x7
        x1_1 = self.down1(x)
        x1_2 = self.conv1(x1_1)
        # branch5x5
        x2_1 = self.down2(x1_1)
        x2_2 = self.conv2(x2_1)
        # branch3x3
        x3_2 = self.down3(x2_1)
        # merge branch1 and branch2
        x3_2_up = F.interpolate(x3_2, (h//4, w//4),
                                mode="bilinear", align_corners=True)
        x2_merge = self.relu(x3_2_up + x2_2)
        x2_up = F.interpolate(x2_merge, (h//2, w//2),
                              mode="bilinear", align_corners=True)
        x1_merge = self.relu(x2_up + x1_2)
        x1_up = F.interpolate(x1_merge, (h, w),
                              mode="bilinear", align_corners=True)
        x1_up = self.conv3(x1_up)
        main_x1_up = x1_up * main_branch_out
        out = self.relu(b1 + main_x1_up)
        return out


class GAUModule(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(GAUModule, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.GroupNorm(32, out_channel)
        )
        self.conv1x1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0),
            nn.GroupNorm(32, out_channel),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1),
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        """
        x is the low level feature
        y is the high level feature
        """
        h, w = x.size(2), x.size(3)
        low_level_feats = self.conv3x3(x)
        global_context = self.conv1x1(y)
        weighted_feats = low_level_feats * global_context
        high_level_feats = F.interpolate(y, (h, w),
                                         mode="bilinear",
                                         align_corners=True)
        out = weighted_feats + high_level_feats
        return out


class PANModule(nn.Module):

    def __init__(self, in_channels_list, out_channel):
        super(PANModule, self).__init__()
        # self.conv3x3 = nn.ModuleList()
        # for i in range(len(in_channels_list)):
        #     self.conv3x3.append(
        #         nn.Sequential(
        #             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
        #             nn.GroupNorm(32, out_channel)
        #         )
        #     )
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight, a=1)
        #         nn.init.constant_(m.bias, 0)
        self.fpa = FPAModule(in_channels_list[-1], out_channel)
        self.gau3 = GAUModule(in_channels_list[-2], out_channel)
        self.gau2 = GAUModule(in_channels_list[-3], out_channel)
        self.gau1 = GAUModule(in_channels_list[0], out_channel)

    def get_result_from_layer_blocks(self, x, idx):
        num_blocks = 0
        for m in self.conv3x3:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.conv3x3:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        names = list(x.keys())
        x = list(x.values())
        results = []

        P5 = self.fpa(x[-1])
        # P5 = self.get_result_from_layer_blocks(P5, -1)
        results.append(P5)
        gau3_out = self.gau3(x[-2], P5)
        P5_up = F.interpolate(P5, (x[-2].size(2), x[-2].size(3)),
                              mode="bilinear", align_corners=True)
        P4 = P5_up + gau3_out
        # P4 = self.get_result_from_layer_blocks(P4, -2)
        results.insert(0, P4)
        P4_up = F.interpolate(P4, (x[-3].size(2), x[-3].size(3)),
                              mode="bilinear", align_corners=True)
        gau2_out = self.gau2(x[-3], P4_up)
        P3 = P4_up + gau2_out
        # P3 = self.get_result_from_layer_blocks(P3, -3)
        results.insert(0, P3)
        P3_up = F.interpolate(P3, (x[0].size(2), x[0].size(3)),
                              mode="bilinear", align_corners=True)
        gau1_out = self.gau1(x[0], P3_up)
        P2 = P3_up + gau1_out
        # P2 = self.get_result_from_layer_blocks(P2, 0)
        results.insert(0, P2)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


if __name__ == "__main__":
    # import torch
    # from layergetter import IntermediateLayerGetter
    # from torchvision import models
    # from torchvision.models._utils import IntermediateLayerGetter as Getter
    # from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as Feat
    # from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


    # image = torch.randn(1, 3, 224, 224)
    # model = models.resnet50(pretrained=True)

    # return_layers = {"layer1": "feat1", "layer2": "feat2",
    #                  "layer3": "feat3", "layer4": "feat4"}
    # return_layers2 = {"layer1": "feat1", "layer2": "feat2",
    #                  "layer3": "feat3", "layer4": "feat4"}
    # body = IntermediateLayerGetter(model, return_layers)
    # fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, MaxpoolOnP5())
    # x = body(image)
    # out = fpn(x)
    # bottom_up = Bottom_up_path([256,256,256,256], 256, MaxpoolOnP5())
    # out = bottom_up(out)
    # import ipdb;ipdb.set_trace()
    # body2 = Getter(model, return_layers2)
    # fpn2 = Feat([256, 512, 1024, 2048], 256, LastLevelMaxPool())
    # x2 = body2(image)
    # out2 = fpn2(x2)
    # body3 = Getter(model, return_layers2)
    # fpn3 = Feat([256, 512, 1024, 2048], 256, LastLevelMaxPool())
    # x3 = body3(image)
    # out3 = fpn3(x2)
    # import ipdb;ipdb.set_trace()
    C2 = torch.randn(1, 256, 224, 224)
    C3 = torch.randn(1, 512, 112, 112)
    C4 = torch.randn(1, 1024, 56, 56)
    C5 = torch.randn(1, 2048, 28, 28)
    x = {"C2": C2, "C3": C3, "C4": C4, "C5": C5}
    pan = PANModule([256, 512, 1024, 2048], 256)
    out = pan(x)
    bottom_up = Bottom_up_path([256,256,256,256], 256, MaxpoolOnP5())
    out = bottom_up(out)
    import ipdb;ipdb.set_trace()



