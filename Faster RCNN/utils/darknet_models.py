from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .darknet_utils.parse_config import *
from .darknet_utils.utils import *
from .darknet_utils.Mish.mish import Mish
from .darknet_utils.FocalLoss import BCEFocalLoss
from .darknet_utils.straightResModule import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs, use_mish=False, old_version=False):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    anchor_scale = (1, 1)
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            ###############
            if not old_version:
                if filters == 18:
                    filters = 21
                if filters == 12:
                    filters = 14
            ###############
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            # pad = int(module_def['pad'])
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                    groups=int(module_def['groups']) if 'groups' in module_def else 1,
                ),
            )


            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"leaky_{module_i}", Mish())

        elif module_def["type"] == "sResMB":
            bn = int(module_def["batch_normalize"]) if 'batch_normalize' in module_def else 0
            filters = int(module_def["filters"])
            pad = int(module_def['pad'])
            ###############
            if not old_version:
                if filters == 18:
                    filters = 21
                if filters == 12:
                    filters = 14
            ###############
            modules.add_module(
                f"sRMB_{module_i}",
                sResModuleBlock(
                    block_id=module_i,
                    ind_num=int(module_def["ind_num"]),
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    pad=pad,
                    stride=int(module_def["stride"]),
                    bn=bn,
                ),
            )

            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif use_mish:
                modules.add_module(f"leaky_{module_i}", Mish())

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            if 'stride' in module_def:
                upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            elif 'nopad' in module_def:
                upsample = NopadUpsample(mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = max(output_filters[1:][int(module_def["from"])], output_filters[-1])
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
            elif module_def["activation"] == "mish":
                modules.add_module(f"leaky_{module_i}", Mish())

        elif module_def["type"] == "reorg3d":
            stride = int(module_def["stride"])
            filters = output_filters[-1]*stride*stride
            modules.add_module(f"reorg3d_{module_i}", EmptyLayer())

        elif module_def['type'] == 'crop':
            filters = output_filters[-1]
            modules.add_module(f"crop2d_{module_i}", EmptyLayer())

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class NopadUpsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, mode="nearest"):
        super(NopadUpsample, self).__init__()
        self.mode = mode

    def forward(self, x, img_dim=None):
        img_dim = max(img_dim)
        scale_factor = (2*img_dim - 128) / (img_dim - 320)
        x = F.interpolate(x, scale_factor=scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path=r"D:\WSI_analysis\det_train\AttFPN-Ovarian-Cancer-master\utils\darknet_config\yolco.cfg", use_mish=False, old_version=False, lite_mode=True, debug_mode=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs, use_mish=use_mish,
                                                            old_version=old_version)
        self.lite_mode = lite_mode
        self.debug_mode = debug_mode
        self.rem_layers = []
        for i, module_def in enumerate(self.module_defs):
            if module_def["type"] == "route":
                for layer_i in module_def["layers"].split(","):
                    layer_i = int(layer_i)
                    if layer_i > 0:
                        self.rem_layers.append(layer_i)
                    else:
                        self.rem_layers.append(i + layer_i)
            elif module_def["type"] == "shortcut":
                self.rem_layers.append(i + int(module_def["from"]))
        self.out_channels = 768

    def forward(self, x, targets=None, fp_flag=False, x_pre_filters="21"):
        if self.lite_mode is None: self.lite_mode = False
        if self.debug_mode is None: self.debug_mode = False
        img_dim = x.shape[2:4]
        d = max(img_dim)

        loss = 0
        layer_outputs, feature_maps, yolos = [], [], []
        first_feature = True
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            if module_def["type"] == "convolutional":
                if module_def["filters"] == x_pre_filters:
                    if first_feature:
                        x = F.interpolate(x, scale_factor=2, mode='nearest')
                        first_feature = False
                        feature_maps.append(x)
                    else:
                        feature_maps.append(x)
                    layer_outputs.append([])
                    continue
            if module_def["type"] in ["convolutional", "maxpool", "sResMB"]:
                x = module(x)
            elif module_def['type'] == 'upsample':
                if 'nopad' in module_def:
                    stage4_size = int(d/16 - 4)
                    x = F.interpolate(x, size=stage4_size, mode='nearest')
                else:
                    x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1]
                a = layer_outputs[layer_i]
                if self.lite_mode:
                    if x == [] or a == []:
                        print("Opps, found BUG in rem_layers: ", self.rem_layers)
                        exit()
                nx = x.shape[1]
                na = a.shape[1]
                if nx == na:
                    x = x + a
                elif nx > na:
                    x[:, :na] = x[:, :na] + a
                else:
                    x = x + a[:, :nx]
                x = module(x)
            if self.lite_mode:
                if i in self.rem_layers:
                    layer_outputs.append(x)
                else:
                    layer_outputs.append([])
            else:
                layer_outputs.append(x)
            if self.debug_mode:
                print(module_def['type'], ": ", x.shape)
                print("Press a key to continue...")
                input()
        return torch.cat(feature_maps, dim=1)
        
