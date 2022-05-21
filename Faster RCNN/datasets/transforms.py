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

class RandomCrop(object):

    def __init__(self, side=1536, crop_size=1024, datatype='train'):
        self.side = side
        self.crop_size = crop_size
        self.datatype = datatype

    def __call__(self, image, target=None):
        old_side = self.side
        new_side = self.crop_size

        if target == None and self.datatype == 'val':
            non_box = True
            crop_x = int(old_side/2) - int(new_side/2)
            crop_y = int(old_side/2) - int(new_side/2)
        else:
            non_box = len(target["boxes"]) == 0

        # if non_box:
        #     pass
        if self.datatype == 'train':
        #     # crop_x = np.random.randint(0, min([old_side - new_side, target["boxes"][:, 0].min()]))
        #     # crop_y = np.random.randint(0, min([old_side - new_side, target["boxes"][:, 1].min()]))
            crop_x = np.random.randint(0, old_side - new_side)
            crop_y = np.random.randint(0, old_side - new_side)
            
        image = np.array(image)
        if self.datatype == 'train':
            if not non_box:
                np_labels = target["labels"].numpy()
                boxes = []
                labels = []
                for bi, (x1, y1, x2, y2) in enumerate(target["boxes"].numpy()):
                    if not is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
                        continue
                    boxes.append([x1-crop_x, y1-crop_y, x2-crop_x, y2-crop_y])
                    labels.append(np_labels[bi])
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
            bbs_ele = BoundingBox(x1=box[0], y1=box[1],
                                  x2=box[2], y2=box[3])
            bbs_list.append(bbs_ele)
        bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
        seq_det = self.aug.to_deterministic()
        image, boxes = seq_det(image=image, bounding_boxes=bbs)
        boxes = np.array([[i.x1, i.y1, i.x2, i.y2]
                          for i in boxes.bounding_boxes])
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

    def __call__(self, image, target):
        image = F.normalize(image,
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
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

