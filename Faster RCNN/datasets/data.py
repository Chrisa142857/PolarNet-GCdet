"""
Faster R-CNN
Building dataset from csv file
The csv file just like
========================================================
image_path    annotation
path/to/img1,0 12 34 20 45;0 23 45 90 100
path/to/img2,0 20 14 59 109
path/to/img3,
path/to/img4,0 19 20 45 29;0 11 34 56 90;0 23 45 34 90
========================================================
0 represents that the instance is a rectangle box; each line a image

copyright (c) Harbin Medical University
Licensed under MIT license
written by Lei Cao
"""
import torch
from PIL import Image
import pandas as pd
import numpy as np


def is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
    new_x1 = x1 - crop_x
    new_x2 = x2 - crop_x
    new_y1 = y1 - crop_y
    new_y2 = y2 - crop_y
    if (new_x1 <= 0 or new_x2 <= 0): return False
    if (new_y1 <= 0 or new_y2 <= 0): return False
    if (new_x1 >= new_side or new_x2 >= new_side): return False
    if (new_y1 >= new_side or new_y2 >= new_side): return False
    return True


class TCTDataset(object):
    """
    Build the TCTDataset from csv file
    """

    def __init__(self, datatype="train", transform=None, labels_dict={}):
        """
        datatype: "train", "val", or "test"
        transform: whether to do transform on image
        labels_dict: a little diff for train and val/test,\
                for train: just like this\
                {"path/to/img1": "0 12 34 23 45;0 123 345 902 454",
                 "path/to/img2": "0 23 45 46 90"}
                for val/test: just like this\
                {"path/to/img5": "",
                 "path/to/img6": "0 441 525 582 636"}
        """
        self.datatype = datatype
        self.transform = transform
        self.labels_dict = labels_dict
        self.image_files_list = list(self.labels_dict.keys())
        self.annotations = [labels_dict[i] for i in self.image_files_list]

    def __getitem__(self, idx):
        # load image
        img_path = self.image_files_list[idx]
        img = Image.open(img_path).convert("RGB")
        annotation = self.labels_dict[img_path]

        boxes = []
        labels = []
        if type(annotation) == str:
            annotation_list = annotation.split(";")
            for anno in annotation_list:
                if len(anno) == 0: continue
                labels.append(int(anno[0])+1)
                anno = anno[2:]  # one box coord str
                anno = anno.split(" ")
                x = [float(anno[0]), float(anno[2])]
                y = [float(anno[1]), float(anno[3])]
                boxes.append([x[0], y[0], x[1], y[1]])

        # convert anything to torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32).clamp(min=0)
        # there is only one class
        # labels = torch.ones((len(boxes),), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # image name
        image_name = torch.tensor(
            [ord(i) for i in list(img_path)],
            dtype=torch.int8
        )
        # make annos on the image into target
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["name"] = image_name

        if self.datatype == "train":
            if self.transform is not None:
                img, target = self.transform(img, target)

            return img, target

        if self.datatype in ["val", "test"]:
            if len(target["boxes"]) != 0:
                np_labels = target["labels"].numpy()
                old_side = 1536
                new_side = 1024
                crop_x = int(old_side/2) - int(new_side/2)
                crop_y = int(old_side/2) - int(new_side/2)
                boxes = []
                labels = []
                for bi, (x1, y1, x2, y2) in enumerate(target["boxes"].numpy()):
                    if not is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
                        continue
                    boxes.append([x1-crop_x, y1-crop_y, x2-crop_x, y2-crop_y])
                    labels.append(np_labels[bi])
                boxes = np.array(boxes)
                labels = np.array(labels)
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            if self.labels_dict[img_path] == "":
                label = 0
            else:
                label = 1
                
            if self.transform is not None:
                img = self.transform(img)

            return img, label, target

    def __len__(self):
        return len(self.image_files_list)


def get_dataset(label_csv_file, datatype, transform):
    """
    Prepare dataset
    label_csv_file: csv file containing image path and annotation
    datatype: "train", "val", or "test"
    transform: transform done on images and targets
    """
    labels_df = pd.read_csv(label_csv_file, na_filter=False)
    if datatype == "train":
        # labels_df = labels_df.loc[
        #     labels_df["annotation"].astype(bool)
        # ].reset_index(drop=True)
        labels_df = labels_df.reset_index(drop=True)
    img_class_dict = dict(zip(labels_df.image_path, labels_df.annotation))
    dataset = TCTDataset(datatype=datatype, transform=transform,
                         labels_dict=img_class_dict)
    return dataset


if __name__ == "__main__":
    train_csv = "../statistic_description/tmp/train.csv"
    val_csv = "../statistic_description/tmp/val.csv"
    dataset_tr = get_dataset(train_csv,
                             datatype="train", transform=None)
    dataset_val = get_dataset(val_csv,
                              datatype="val", transform=None)

    import ipdb;ipdb.set_trace()

