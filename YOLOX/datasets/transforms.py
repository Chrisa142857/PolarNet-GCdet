#!/usr/bin/env python
# coding=utf-8
"""
可以在这里写一些数据增强的函数
"""
import random

import numpy as np
import torch
from torchvision.transforms import functional as F
import PIL
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from datasets.data import is_box_valid_after_crop


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[0] = x[0] * std[0] + mean[0]
    x[1] = x[1] * std[1] + mean[1]
    x[2] = x[2] * std[2] + mean[2]
    return x

class XyxyToXywh(object):
    
    def __init__(self):
        pass

    def __call__(self, image, target):
        xyxy = target['boxes']
        if 0 in xyxy.shape:
            return image, target
        xywh = torch.zeros_like(xyxy)
        xywh[:, :2] = xyxy[:, :2] / 2 + xyxy[:, 2:] / 2
        xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
        target['boxes'] = xywh
        return image, target

class RandomCrop(object):

    def __init__(self, side=1536, crop_size=1024, datatype='train', ignore=False):
        self.side = side
        self.crop_size = crop_size
        self.datatype = datatype
        self.ignore = ignore

    def __call__(self, image, target=None):
        old_side = self.side
        new_side = self.crop_size
        if self.ignore:
            crop_x = 0
            crop_y = 0
        elif self.datatype == 'train':
            crop_x = np.random.randint(0, old_side - new_side)
            crop_y = np.random.randint(0, old_side - new_side)
        else:
            crop_x = int(old_side/2) - int(new_side/2)
            crop_y = int(old_side/2) - int(new_side/2)

        image = np.array(image)
        if self.datatype == 'train':
            np_labels = target["labels"].numpy()
            boxes = []
            labels = []
            for bi, (x1, y1, x2, y2) in enumerate(target["boxes"].numpy()):
                if not is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
                    continue
                boxes.append([x1-crop_x, y1-crop_y, x2-crop_x, y2-crop_y])
                labels.append([np_labels[bi]])
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            return image[crop_y:crop_y+new_side, crop_x:crop_x+new_side], target
        else:
            return image[crop_y:crop_y+new_side, crop_x:crop_x+new_side]


class ImgAugTransform(object):
    """
    Use imgaug package to do data augmentation
    """
    def __init__(self):
        self.aug = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])

    def __call__(self, image, target):
        image = np.array(image)
        bbs_list = []
        boxes = target["boxes"].numpy()
        for box in boxes:
            # bbs_ele = BoundingBox(x1=box[0]-box[2]/2, y1=box[1]-box[3]/2,
            #                       x2=box[1]+box[2]/2, y2=box[2]+box[3]/2)
            bbs_ele = BoundingBox(x1=box[0], y1=box[1],
                                  x2=box[2], y2=box[3])
            bbs_list.append(bbs_ele)
        bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
        seq_det = self.aug.to_deterministic()
        image, boxes = seq_det(image=image, bounding_boxes=bbs)
        b = []
        for i in boxes.bounding_boxes:
            # b.append([i.x1/2 + i.x2/2, i.y1/2 + i.y2/2, i.x2 - i.x1, i.y2 - i.y1])
            b.append([i.x1, i.y1, i.x2, i.y2])
        boxes = np.array(b)
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        return PIL.Image.fromarray(image), target


class Compose(object):
    """
    作用:
        将transforms整合在一起
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    """
    将PILImage变成张量
    """

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    """
    Modified Normalize
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image,
                            mean=self.mean,
                            std=self.std)
        return image, target


class RandomHorizontalFlip(object):
    """
    作用:
        进行水平翻转
    参数:
        prob: 进行随机翻转的数据比例
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 因为图像为3D，W在最后一个维度
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomVerticalFlip(object):
    """
    作用:
        进行垂直翻转数据增强
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(1)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target

