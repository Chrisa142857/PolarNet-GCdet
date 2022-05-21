#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from torch.jit.annotations import Dict, List, Tuple, Optional
from torch import Tensor

from collections import OrderedDict
import warnings


class GeneralizedRCNN(nn.Module):

    def __init__(self, transform, backbone, polar_net, rpn, roi_heads, mtype=None):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.polar_net = polar_net
        if 'a' in mtype: 
            self.feat_k = '2'
        elif 'b' in mtype: 
            self.feat_k = '1'
        elif 'c' in mtype: 
            self.feat_k = '0'
        else: 
            self.feat_k = '3'
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False
        self.trace_module = False
        self.extracting_polarity = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training and not self.extracting_polarity:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        if self.training and targets is None:
            raise ValueError("Targets should be passed during training time")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            img_size = img.shape[-2:]
            assert len(img_size) == 2
            original_image_sizes.append((img_size[0], img_size[1]))
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, Tensor):
            # assert features.shape[1] == self.polar_net.hidden_num
            # polar_losses, features = self.polar_net(features)
            features = OrderedDict([("0", features)])
        else:
            polarity, features[self.feat_k] = self.polar_net(features[self.feat_k])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes,
                                                     targets=targets,
                                                     polarity=polarity, polar_net=self.polar_net, topk=10)
        
        detections = self.transform.postprocess(detections,
                                                images.image_sizes,
                                                original_image_sizes)
        losses = {}
        # losses.update(polar_losses)
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.trace_module:
            return detections
        elif torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (loss, detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)

