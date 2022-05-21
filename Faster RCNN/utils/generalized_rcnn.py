#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from torch.jit.annotations import Dict, List, Tuple, Optional
from torch import Tensor

from collections import OrderedDict
import warnings


class GeneralizedRCNN(nn.Module):

    def __init__(self, transform, backbone, rpn, roi_heads):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False
        self.trace_module = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
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
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes,
                                                     targets)
        detections = self.transform.postprocess(detections,
                                                images.image_sizes,
                                                original_image_sizes)
        losses = {}
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

