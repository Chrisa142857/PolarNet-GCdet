#!/usr/bin/env python
# coding=utf-8
"""
专门针对VOC数据的数据集生成工具
"""
import torch
import torchvision
import os
import pandas as pd
from PIL import Image
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
from tools import transforms as T
from myconfig.config import OPT


class VOCCustomData(torchvision.datasets.vision.VisionDataset):
    """
    数据集生成

    参数:
        root: VOC数据的根目录
        image_set: 要使用的数据集的名称比如train, trainval等
        transform
        target_transform: 函数，对target做相应的处理
        transforms: 自定义的transforms处理函数
    """

    def __init__(self,
                root=OPT.root,
                imgset = OPT.voc_imgset_file,
                transform=None,
                target_transform=None,
                transforms=None,
                dataset_flag='train',
                use_csv=False,
                csv_dir="../csv_tools/new_filter.csv"):
        super(VOCCustomData, self).__init__(
            root, transforms, transform, target_transform
        )
        self.root = root
        self.transforms = transforms
        self.use_csv = use_csv
        self.csv_dir = csv_dir

        voc_root = self.root
        self.voc_imgset_file = imgset  # 生成的txt文件存储位置
        self.image_dir = os.path.join(voc_root, "JPEGImages")
        self.annotation_dir = os.path.join(voc_root, "Annotations")
        if not os.path.isdir(voc_root):
            raise RuntimeError("未发现VOC文件夹，请检查!")

        # 图像名
        file_names = []
        for imgs in os.listdir(self.image_dir):
            file_names.append(imgs.split(".")[0])
        # 这里恰巧少了2张图片，所以将就用
        if use_csv:
            filtered_df = pd.read_csv(self.csv_dir, index_col=0)
            filtered_path = list(filtered_df['0'])
            filtered_file_names = [i.split("/")[-1].strip(".png")
                                  for i in filtered_path]
            filtered_file_names = set(filtered_file_names)
            file_names = list(set(file_names).intersection(filtered_file_names))


        # 划分训练集，验证集和测试集
        train_file_names, val_test_file_names = train_test_split(
            file_names, test_size=OPT.val_ratio * 2,
            random_state=0
        )
        val_file_names, test_file_names = train_test_split(
            val_test_file_names, test_size=0.5,
            random_state=0
        )
        image_file = pd.DataFrame(file_names, index=None)
        train_image_file = pd.DataFrame(train_file_names, index=None)
        val_image_file = pd.DataFrame(val_file_names, index=None)
        test_image_file = pd.DataFrame(test_file_names, index=None)
        # 保存图像名，没有后缀的那种
        image_file.to_csv(
            imgset + "/imageset.txt",
            header=False,
            index=False
        )
        train_image_file.to_csv(
            imgset + "/imageset_train.txt",
            header=False,
            index=False
        )
        val_image_file.to_csv(
            imgset + "/imageset_val.txt",
            header=False,
            index=False
        )
        test_image_file.to_csv(
            imgset + '/imageset_test.txt',
            header=False,
            index=False
        )
        img_extension = OPT.image_format_extension
        file_names_pool = {
            'train': train_file_names,
            'val': val_file_names,
            'test': test_file_names,
        }
        self.images_path = [
            os.path.join(self.image_dir, x + img_extension)
            for x in file_names_pool[dataset_flag]
        ]
        self.anno_path = [
            os.path.join(self.annotation_dir, x + ".xml")
            for x in file_names_pool[dataset_flag]
        ]
        image_set_pool = {
            "train": "imageset_train",
            "val": "imageset_val",
            "test": "imageset_test",
        }
        self.image_set = image_set_pool[dataset_flag]

        assert (len(self.images_path) == len(self.anno_path)), \
                "图像的张数和标注文件的个数不一致，请检查!"

    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert("RGB")
        target = self.parse_voc_xml(
            ET.parse(self.anno_path[index]).getroot()
        )
        target = dict(
            image_id = index,
            annotations = target["doc"]
        )
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_path)

    def parse_voc_xml(self, node):
        """
        解析xml文件，返回xml文件到字典中

        参数:
            node: ET解析到的子节点
        """
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                {ind: v[0] if len(v) == 1 else v
                for ind, v in def_dic.items()}
            }

        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text

        return voc_dict


class ConvertCustomVOCtoCOCO(object):
    """
    转为COCO格式
    """

    def __call__(self, image, target):
        # 标注信息
        anno = target['annotations']
        # 得到图片名
        delimiter = '_'
        filename = delimiter.join(
            anno['path'].split("\\")[-2:]
        ).split(".")[0]
        # 图像的高和宽，确保box不超出图像
        w, h = image.size
        # 存储标签，坐标等
        boxes = []
        classes = []
        ishard = []
        objs = anno['outputs']['object']['item']
        if isinstance(objs, dict):
            objs = [objs]
        try:
            for obj in objs:
                if 'bndbox' in obj:
                    bbox = obj['bndbox']
                    xmin = float(bbox['xmin'])
                    ymin = float(bbox['ymin'])
                    xmax = float(bbox['xmax'])
                    ymax = float(bbox['ymax'])
                    # 框的宽度
                    bbox_ws = xmax - xmin
                    # 框的高度
                    bbox_hs = ymax - ymin
                    # 确保都是大于0的数
                    if bbox_ws > 0 and bbox_hs > 0:
                        boxes.append([xmin, ymin, xmax, ymax])
                        name = obj['name']
                        if name == "异常":
                            name = "Positive"
                        classes.append(OPT.box_label_names.index(name))
                        ishard.append(0)
        except KeyError as e:
            import ipdb;ipdb.set_trace()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        ishard = torch.as_tensor(ishard)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard
        target["name"] = torch.tensor(
            [ord(i) for i in list(filename)], dtype=torch.int8
        )

        return image, target


def get_custom_voc(root, transforms, dataset_flag, use_csv=False):
    """
    将上面的类实例化并组合
    """
    t = [ConvertCustomVOCtoCOCO()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = VOCCustomData(root=root, transforms=transforms,
                           dataset_flag=dataset_flag,
                           use_csv=use_csv)
    return dataset


if __name__ == "__main__":
    dataset = get_custom_voc(root=OPT.root, transforms=None,
                            dataset_flag='train',
                            use_csv=True)
    dataset[3]

