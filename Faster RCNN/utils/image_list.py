#!/usr/bin/env python
# coding=utf-8
import torch
from torch.jit.annotations import List, Tuple


class ImageList(object):

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]])
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

