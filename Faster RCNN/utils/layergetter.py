#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
from torch.jit.annotations import Dict

from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str]
    }

    def __init__(self, model, return_layers):
        ori_return_layers = return_layers.copy()
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = ori_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DenseNetLayerGetter(nn.Module):
    def __init__(self, model):
        super(DenseNetLayerGetter, self).__init__()
        self.model = model
        self.out_dict = OrderedDict()

    def forward(self, x):
        x = self.model.features.conv0(x)
        x = self.model.features.norm0(x)
        x = self.model.features.relu0(x)
        x = self.model.features.pool0(x)
        x = self.model.features.denseblock1(x)
        x = self.model.features.transition1.norm(x)
        x = self.model.features.transition1.relu(x)
        C2 = x = self.model.features.transition1.conv(x)
        self.out_dict["0"] = C2
        x = self.model.features.transition1.pool(x)
        x = self.model.features.denseblock2(x)
        x = self.model.features.transition2.norm(x)
        x = self.model.features.transition2.relu(x)
        C3 = x = self.model.features.transition2.conv(x)
        self.out_dict["1"] = C3
        x = self.model.features.transition2.pool(x)
        x = self.model.features.denseblock3(x)
        x = self.model.features.transition3.norm(x)
        x = self.model.features.transition3.relu(x)
        C4 = x = self.model.features.transition3.conv(x)
        self.out_dict["2"] = C4
        x = self.model.features.transition3.pool(x)
        C5 = x = self.model.features.denseblock4(x)
        self.out_dict["3"] = C5
        return self.out_dict


if __name__ == "__main__":
    from torchvision import models
    import torch
    from torchvision.models._utils import IntermediateLayerGetter as Getter

    x = torch.randn(1, 3, 224, 224)
    model = models.resnet50(pretrained=True)
    return_layers1 = {"layer1": "feat1", "layer2": "feat2"}
    return_layers2 = {"layer1": "feat1", "layer2": "feat2"}

    layergetter = IntermediateLayerGetter(model, return_layers1)
    out = layergetter(x)
    getter = Getter(model, return_layers2)
    out2 = getter(x)
    print((out["feat1"]==out2["feat1"]).all())
    print((out["feat2"]==out2["feat2"]).all())

