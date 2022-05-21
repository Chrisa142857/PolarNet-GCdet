from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile

from utils.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone
from faster_rcnn import FasterRCNN, FastRCNNPredictor
from faster_polar_rcnn import FasterPolarRCNN
from polar_net import PolarNet

def get_model_info(model, tsize):
    device = 'cuda'
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=device) #next(model.parameters()).device
    flops, params = profile(deepcopy(model.to(device)), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


if __name__ == "__main__":

    tsize = (1024, 1024)
    use_attn = 0
    use_polar = 1

    CLASSES = {"__background__", "nGEC", "AGC"}
    backbone = resnet_fpn_backbone("resnet50", False, use_attn=use_attn)
    if use_polar == 1:
        polar_net = PolarNet()
        model = FasterPolarRCNN(backbone, polar_net, num_classes=len(CLASSES))
    else:
        model = FasterRCNN(backbone, num_classes=len(CLASSES))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      len(CLASSES))
    print(get_model_info(model, tsize))