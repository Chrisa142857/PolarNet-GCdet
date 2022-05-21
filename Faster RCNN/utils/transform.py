#!/usr/bin/env python
# coding=utf-8
import torch
from torch import Tensor
import torch.nn as nn
from torch.jit.annotations import List, Tuple, Dict, Optional
import math

from .image_list import ImageList


def resize_image(image, self_min_size, self_max_size, target):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))  # 图像的短边
    max_size = float(torch.max(im_shape))  # 图像的长边
    scale_factor = self_min_size / min_size  # 缩放因子
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear",
        align_corners=False
    )[0]
    if target is None:
        return image, target
    return image, target


def resize_boxes(boxes, original_size, new_size):
    if boxes.shape[0] == 0: return boxes
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_ori, dtype=torch.float32, device=boxes.device)
        for s, s_ori in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
    
    xmin = xmin * ratio_w
    xmax = xmax * ratio_w
    ymin = ymin * ratio_h
    ymax = ymax * ratio_h

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors")
            image = self.normalize(image)
            if targets != None:
                image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
            
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 将图像列表变为4D张量
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image-mean[:, None, None])/std[:, None, None]

    def torch_choice(self, l):
        # type: (List[int])
        index = int(torch.empty(1).uniform_(0., float(len(l))).item())
        return l[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]])
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        image, target = resize_image(image, size,
                                     float(self.max_size), target)
        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int)
        max_size = self.max_by_axis([list(img.shape)
                                     for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1])/stride)*stride)
        max_size[2] = int(math.ceil(float(max_size[2])/stride)*stride)

        # 4D张量的形状
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        num_img = len(images)
        for i in range(num_img):
            img, pad_img = images[i], batched_imgs[i]
        # for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]])
        if self.training:
            return result
        for i, (pred, im_s, o_img_s) in enumerate(zip(
            result, image_shapes, original_image_sizes
        )):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_img_s)
            result[i]["boxes"] = boxes
        return result

