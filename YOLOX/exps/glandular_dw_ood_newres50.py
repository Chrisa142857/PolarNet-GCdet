# encoding: utf-8
from concurrent.futures.process import _threads_wakeups
import os
from re import L

import torch
import torch.distributed as dist
from torchvision import transforms
import time

import pandas as pd
import numpy as np
from collections import namedtuple

import torchvision
from torchvision.transforms.transforms import ToPILImage
from PIL import Image
from yolox.exp import Exp as MyExp
from yolox.models import yolox
from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized
from datasets import transforms as T
from datasets.data import get_dataset, is_box_valid_after_crop, collate_fn
from sklearn.metrics import confusion_matrix
import random
from tqdm import tqdm
from yolox.utils.boxes import bboxes_iou
from yolox.data.data_augment import _mirror, augment_hsv

from datasets.transforms import denormalize

def get_transform(train):
    """
    Data augmentation ops
    train: a boolean flag to do diff transform in train or val, test
    """
    if not train:
        transform_list = [
            T.RandomCrop(datatype='val'),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        return transforms.Compose(transform_list)
    else:
        transform_list = [
            T.RandomCrop(datatype='train'),
            T.ImgAugTransform(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            T.XyxyToXywh()
        ]
        return T.Compose(transform_list)


Object = namedtuple("Object", ["image_path", "object_id", "object_type",
                     "coordinates"])
Prediction = namedtuple("Prediction", ["image_path", "probability", "object_type", "coordinates"])


def voc_ap(recall, precision, use_07_metric=False):
    """
    Calculate the AP value using recall and precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p/11.

    else:
        mrec = np.concatenate(([0.], recall, [1.]))
        mprec = np.concatenate(([0.], precision, [0.]))
        for i in range(mprec.size - 1, 0, -1):
            mprec[i-1] = np.maximum(mprec[i-1], mprec[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1] - mrec[i]) * mprec[i+1])

    return ap

def custom_voc_eval(gt_csv, pred_csv, num_class=2, ovthresh=0.5, use_07_metric=False):
    """
    Do custom eval, include mAP and FROC

    gt_csv: path/to/ground_truth_csv
    pred_csv: path/to/pred_csv
    ovthresh: iou threshold
    """
    # parse ground truth csv, by parsing the ground truth csv,
    # we get ground box info
    old_side = 1536
    new_side = 1024
    crop_x = int(old_side / 2) - int(new_side / 2)
    crop_y = int(old_side / 2) - int(new_side / 2)
    num_image = 0
    num_object = 0
    num_object_subclass = [0 for _ in range(num_class)]
    object_dict = {}
    with open(gt_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, annotation = line.strip("\n").split(",")
            if annotation == "":
                num_image += 1
                continue

            object_annos = annotation.split(";")
            for object_anno in object_annos:
                fields = object_anno.split(" ")  # one box
                if len(fields) == 0 or fields[0] == '':
                    continue
                object_type = int(fields[0])
                coords = np.array(list(map(float, fields[1:])))
                if 0 in coords.shape:
                    continue
                x, y, w, h = coords
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                if not is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
                    continue
                # one box info
                obj = Object(image_path, num_object, object_type, coords)
                if image_path in object_dict:
                    object_dict[image_path].append(obj)
                else:
                    object_dict[image_path] = [obj]
                num_object += 1
                num_object_subclass[int(object_type)] += 1
            num_image += 1

    # parse prediction csv, by parsing pred csv, we get the pre box info
    preds = []
    with open(pred_csv) as f:
        # skip header
        next(f)
        for line in f:
            image_path, prediction = line.strip("\n").split(",")

            if prediction == "":
                continue

            coord_predictions = prediction.split(";")
            for coord_prediction in coord_predictions:
                fields = coord_prediction.split(" ")
                probability, object_type, x1, y1, x2, y2 = list(map(float, fields))
                pred = Prediction(image_path, probability, object_type,
                                  np.array([x1, y1, x2, y2]))
                preds.append(pred)

    # sort prediction by probability, decrease order
    preds = sorted(preds, key=lambda x: x.probability, reverse=True)
    nd = len(preds)  # total number of pred boxes
    object_hitted = set()
    object_hitted_nonclass = set()
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_nonclass = np.zeros(nd)
    fp_nonclass = np.zeros(nd)
    tp_subclass = [np.zeros(nd) for _ in range(num_class)]
    fp_subclass = [np.zeros(nd) for _ in range(num_class)]

    # loop over each pred box to see if it matches one ground box
    if nd > 0:
        for d in range(nd):
            if preds[d].image_path in object_dict:
                # one pred box coords
                bb = preds[d].coordinates.astype(float)
                pred_type = preds[d].object_type
                image_path = preds[d].image_path
                # set the initial max overlap iou
                ovmax = -np.inf
                # ground box on the image
                R = [i.coordinates for i in object_dict[image_path] if 0 not in i.coordinates.shape]
                try:
                    BBGT = np.stack(R, axis=0)
                except ValueError:
                    import ipdb;
                    ipdb.set_trace()
                R_img_id = [i.object_id for i in object_dict[image_path]]
                R_object_type = [i.object_type for i in object_dict[image_path]]
                BBGT_hitted_flag = np.stack(R_img_id, axis=0)
                BBGT_object_type = np.stack(R_object_type, axis=0)

                if BBGT.size > 0:
                    # cal the iou between pred box and all the gt boxes on
                    # the image
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])

                    # cal inter area width
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih  # inter area

                    # cal iou
                    union = (
                            (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                            (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - \
                            inters
                    )
                    overlaps = inters / union
                    # find the max iou
                    ovmax = np.max(overlaps)
                    # find the index of the max iou
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    # if the max iou greater than the iou thresh
                    if BBGT_hitted_flag[jmax] not in object_hitted_nonclass:
                        tp_nonclass[d] = 1.
                        object_hitted_nonclass.add(BBGT_hitted_flag[jmax])
                    else:
                        fp_nonclass[d] = 1.
                    if BBGT_hitted_flag[jmax] not in object_hitted and BBGT_object_type[jmax] == pred_type:
                        tp_subclass[int(pred_type)][d] = 1.
                        tp[d] = 1.
                        object_hitted.add(BBGT_hitted_flag[jmax])
                    else:
                        fp_subclass[int(pred_type)][d] = 1.
                        fp[d] = 1.
                else:
                    fp_subclass[int(pred_type)][d] = 1.
                    fp_nonclass[d] = 1.
                    fp[d] = 1.
            else:
                # fp[d] = 1.
                continue

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    fp_nonclass = np.cumsum(fp_nonclass)
    tp_nonclass = np.cumsum(tp_nonclass)
    fp_subclass = [np.cumsum(sub) for sub in fp_subclass]
    tp_subclass = [np.cumsum(sub) for sub in tp_subclass]

    recs, precs, aps = {}, {}, {}

    rec = tp / float(num_object)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    recs['all_%d'%(ovthresh*100)] = rec
    precs['all_%d'%(ovthresh*100)] = prec
    aps['all_%d'%(ovthresh*100)] = ap
    rec = tp_nonclass / float(num_object)
    prec = tp_nonclass / np.maximum(tp_nonclass + fp_nonclass, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    recs['nonclass_%d'%(ovthresh*100)] = rec
    precs['nonclass_%d'%(ovthresh*100)] = prec
    aps['nonclass_%d'%(ovthresh*100)] = ap

    for i, (tp, fp, num_object) in enumerate(zip(tp_subclass, fp_subclass, num_object_subclass)):
        rec = tp / float(num_object)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        recs['class%d_%d'%(i, ovthresh*100)] = rec
        precs['class%d_%d'%(i, ovthresh*100)] = prec
        aps['class%d_%d'%(i, ovthresh*100)] = ap


    return recs, precs, aps


class orig_glandular_evaluator:

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        ood_iou_thres,
        ood_input_side,
        gt_csv_path='datasets/val_glandular.csv',
        cls_csv_path='temp_cls.csv',
        loc_csv_path='temp_loc.csv'
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)
        self.gt_csv_path = gt_csv_path
        self.cls_csv_path = cls_csv_path
        self.loc_csv_path = loc_csv_path
        self.ood_iou_thres = ood_iou_thres
        self.ood_input_side = ood_input_side

    @torch.no_grad()
    def evaluate(
        self,
        model,
        score_thres=0.1,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        progress_bar = tqdm
        old_side = 1536
        new_side = 1024
        crop_x = int(old_side / 2) - int(new_side / 2)
        crop_y = int(old_side / 2) - int(new_side / 2)
        if getattr(model, "eval_preds", None) is not None:
            model.eval_preds = []
            model.eval_labels = []
        model.eval()

        preds = []
        locs = []
        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            imgs = imgs.type(torch.cuda.FloatTensor)

            is_time_record = cur_iter < len(self.dataloader) - 1

            if is_time_record:
                start = time.time()
            outputs = model(imgs, target=targets, yolox_inference=True)

            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start
            
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

            if is_time_record:
                nms_end = time_synchronized()
                nms_time += nms_end - infer_end

            if outputs[-1] == None:
                locs.append("")
                preds.append(0)
                continue

            output = outputs[-1].cpu()

            bboxes = output[:, 0:4]

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            if np.random.randint(1, 100) <= 5:
                val_save_img = imgs[0].detach().cpu()
                val_save_label = targets[0].detach().cpu()
                val_save_pred_box = bboxes
                val_save_pred_cls = cls
                val_save_pred_score = scores


            coords = []  # used to save pred coords x1 y1 x2 y2
            coords_score = []  # used to save pred box scores
            coords_label = []  # used to save pred box label
            if len(bboxes) == 0:
                # if no pred boxes, means that the image is negative
                preds.append(0)
                coords.append([])
                coords_score.append("")
                locs.append("")
                continue
            else:
                # we keep those pred boxes whose score is more than 0.1
                new_output_index = torch.where(scores >= score_thres)
                new_boxes = bboxes[new_output_index]
                new_scores = scores[new_output_index]
                new_cls = cls[new_output_index]
                if len(new_boxes) != 0:
                    preds.append(torch.max(new_scores).tolist())
                else:
                    preds.append(0)

                for i in range(len(new_boxes)):
                    new_box = new_boxes[i].tolist()
                    coords.append([new_box[0], new_box[1],
                                   new_box[2], new_box[3]])
                coords_score += new_scores.tolist()
                coords_label += new_cls.tolist()
                line = ""
                for i in range(len(new_boxes)):
                    if i == len(new_boxes) - 1:
                        line += str(coords_score[i]) + ' ' + str(int(coords_label[i])) + ' ' + str(
                            coords[i][0] + crop_x) + ' ' + \
                                str(coords[i][1] + crop_y) + ' ' + str(coords[i][2] + crop_x) + ' ' + \
                                str(coords[i][3] + crop_y)
                    else:
                        line += str(coords_score[i]) + ' ' + str(int(coords_label[i])) + ' ' + str(
                            coords[i][0] + crop_x) + ' ' + \
                                str(coords[i][1] + crop_y) + ' ' + str(coords[i][2] + crop_x) + ' ' + \
                                str(coords[i][3] + crop_y) + ';'

                locs.append(line)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"
        cls_res = pd.DataFrame(
            {"image_path": self.dataloader.dataset.image_files_list,
             "prediction": preds}
        )
        print("====write cls pred results to csv====")
        cls_res.to_csv(self.cls_csv_path, columns=["image_path", "prediction"],
                       sep=',', index=None)
        loc_res = pd.DataFrame(
            {"image_path": self.dataloader.dataset.image_files_list,
             "prediction": locs}
        )
        print("====write loc pred results to csv====")
        loc_res.to_csv(self.loc_csv_path, columns=["image_path", "prediction"],
                       sep=',', index=None)
        # gt_anno = pd.read_csv(gt_csv_path, na_filter=False)
        # gt_label = gt_anno.annotation.astype(bool).astype(float).values
        # pred = cls_res.prediction.values
        aps = []
        for i in range(1, 10):
            recall, precision, ap = custom_voc_eval(self.gt_csv_path, self.loc_csv_path, ovthresh=i / 10, num_class=2)
            aps += [ap]
            
        if getattr(model, "eval_preds", None) is not None:
            result = {k.replace('all', 'al'): ap[k] for ap in aps for k in ap}
            ood_preds = torch.stack(model.eval_preds).squeeze()
            ood_labels = torch.cat(model.eval_labels)
            is_ood = ood_preds == ood_labels
            result['all_ood_acc'] = torch.sum(is_ood) / len(is_ood)
            m = confusion_matrix(ood_labels, ood_preds)
            result['confusion_matrix[0, 0]'] = m[0][0]
            result['confusion_matrix[0, 1]'] = m[0][1]
            result['confusion_matrix[1, 0]'] = m[1][0]
            result['confusion_matrix[1, 1]'] = m[1][1]
            tp_rec = result['confusion_matrix[1, 1]'] / (result['confusion_matrix[1, 0]'] + result['confusion_matrix[1, 1]'])
            fp_rec = result['confusion_matrix[0, 0]'] / (result['confusion_matrix[0, 0]'] + result['confusion_matrix[0, 1]'])
            result['tp_rec'] = tp_rec
            result['fp_rec'] = fp_rec
        else:
            result = {k: ap[k] for ap in aps for k in ap}
        result['val_save_img'] = val_save_img
        result['val_save_label'] = val_save_label
        result['val_save_pred_box'] = val_save_pred_box
        result['val_save_pred_cls'] = val_save_pred_cls
        result['val_save_pred_score'] = val_save_pred_score
        return result, info


def shift_box_in_image(x1, y1, x2, y2, iw, ih, box_side):
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    x1 = max(x1, 0)
    x2 = x1 + box_side
    y1 = max(y1, 0)
    y2 = y1 + box_side
    x2 = min(x2, iw)
    x1 = x2 - box_side
    y2 = min(y2, ih)
    y1 = y2 - box_side
    return [
        x1, y1,
        x2, y2
        ]


class glandular_evaluator:

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        ood_iou_thres,
        ood_input_side,
        gt_csv_path='datasets/val_glandular.csv',
        cls_csv_path='temp_cls.csv',
        loc_csv_path='temp_loc.csv'
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)
        self.gt_csv_path = gt_csv_path
        self.cls_csv_path = cls_csv_path
        self.loc_csv_path = loc_csv_path
        self.ood_iou_thres = ood_iou_thres
        self.ood_input_side = ood_input_side

    @torch.no_grad()
    def evaluate(
        self,
        model,
        score_thres=0.1,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        progress_bar = tqdm
        if getattr(model, "eval_preds", None) is not None:
            model.eval_preds = []
            model.eval_labels = []
        model.eval()

        eval_preds = []
        eval_labels = []
        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            imgs = imgs.type(torch.cuda.FloatTensor)

            is_time_record = cur_iter < len(self.dataloader) - 1

            if is_time_record:
                start = time.time()
            outputs = model(imgs, target=None, yolox_inference=False)

            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start
            

            if is_time_record:
                nms_end = time_synchronized()
                nms_time += nms_end - infer_end

            _, eval_pred = outputs.max(dim=-1)
            eval_label = targets
            eval_preds.append(eval_pred)
            eval_labels.append(eval_label)


        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"
        
        result = {}
        ood_preds = torch.stack(eval_preds).squeeze().cpu()
        ood_labels = torch.cat(eval_labels).cpu()
        is_ood = ood_preds == ood_labels
        result['all_ood_acc'] = torch.sum(is_ood) / len(is_ood)
        m = confusion_matrix(ood_labels, ood_preds)
        result['confusion_matrix[0, 0]'] = m[0][0]
        result['confusion_matrix[0, 1]'] = m[0][1]
        result['confusion_matrix[1, 0]'] = m[1][0]
        result['confusion_matrix[1, 1]'] = m[1][1]
        tp_rec = result['confusion_matrix[1, 1]'] / (result['confusion_matrix[1, 0]'] + result['confusion_matrix[1, 1]'])
        fp_rec = result['confusion_matrix[0, 0]'] / (result['confusion_matrix[0, 0]'] + result['confusion_matrix[0, 1]'])
        result['tp_rec'] = tp_rec
        result['fp_rec'] = fp_rec

        return result, info

class OODDataset(object):

    def __init__(self, orig_img_list=None, bbox_list=None, label_list=None, input_side=None, tile_list=None) -> None:
        self.orig_img_list = orig_img_list
        # self.tile_list = torch.stack(tile_list)
        self.tile_list = tile_list
        assert self.orig_img_list!=None or self.tile_list!=None
        if self.orig_img_list != None:
            self.image_id = []
            for i in range(len(orig_img_list)):
                self.image_id += [i for _ in range(len(bbox_list[i]))]
            self.boxes = torch.cat(bbox_list, dim=0)
        self.labels = torch.cat(label_list, dim=0)
        self.tp_idx = torch.where(self.labels == 1)[0].tolist()
        self.fp_idx = torch.where(self.labels == 0)[0].tolist()
        self.sample_len = min(len(self.tp_idx), len(self.fp_idx))
        self.shuffle_idx()
        self.input_side = input_side
        self.to_pil = ToPILImage()
        self.to_tensor = T.ToTensor()
        self.normalizer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        self.is_eval = False
        print("OOD data num:", len(self.labels))
        print("OOD TP num:", self.labels.sum())
        print("OOD FP num:", len(self.labels)-self.labels.sum())
    
    def shuffle_idx(self):
        random.shuffle(self.tp_idx)
        random.shuffle(self.fp_idx)
        self.idx = self.tp_idx[:self.sample_len] + self.fp_idx[:self.sample_len]

    def aug_img(self, tile):
        tile = self.to_pil(tile)
        tile = np.array(tile)
        tile, _ = _mirror(tile)
        tile = Image.fromarray(tile.astype('uint8')).convert('RGB')
        tile, _ = self.to_tensor(tile, None)
        return tile

    def __getitem__(self, index):
        index = self.idx[index]
        label = self.labels[index]
        tile = self.tile_list[index]
        if self.input_side and (tile.shape[-1] != self.input_side or tile.shape[-2] != self.input_side):
            tile = torch.nn.functional.interpolate(tile.unsqueeze(0), size=self.input_side)[0]

        # if not self.is_eval:
        #     tile = self.aug_img(tile)
        return tile, label, torch.as_tensor([tile.shape[-2], tile.shape[-1]]), torch.as_tensor([index])
    
    def __len__(self):
        return self.sample_len*2


import cv2

class StructuredPool(torch.nn.Module):
    def __init__(self, gsize=25, tshape=(8, 8)):
        super(StructuredPool, self).__init__()
        pixel_num = tshape[0] * tshape[1]
        self.tshape = tshape
        self.gsize = gsize
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(pixel_num, pixel_num, True)
        self.unflatten = torch.nn.Unflatten(-1, tshape)
        self.avgpool = torch.nn.AvgPool2d(tshape)
        self.to_pil = transforms.ToPILImage()
    
    def denormalize(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        x[0] = x[0] * std[0] + mean[0]
        x[1] = x[1] * std[1] + mean[1]
        x[2] = x[2] * std[2] + mean[2]
        return x
    
    def build_mask(self, torch_imgs):
        self.mask = torch.zeros_like(torch_imgs)[:, 0, :, :]
        for i in range(self.mask.shape[0]):
            img = np.array(self.to_pil(denormalize(torch_imgs[i].clone())))
            hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            S_img = hls_img[..., 2]
            I_img = hls_img[..., 1]
            info = np.iinfo(S_img.dtype) # Get the information of the incoming image type 
            S_img = S_img.astype(np.float64)/info.max
            S_img = S_img * ((S_img - S_img.min()) / (S_img.max() - S_img.min()))
            S_img = (S_img * info.max).astype(np.uint8)
            blur_simg = cv2.GaussianBlur(S_img, (self.gsize, self.gsize), 0)
            blur_iimg = cv2.GaussianBlur(I_img, (self.gsize, self.gsize), 0)
            sub_img = blur_simg - blur_iimg
            self.mask[i] = torch.from_numpy(sub_img) / 255
        self.mask = self.mask.clamp(min=0)

    def forward(self, x):
        test_phase = self.tshape[0] != x.shape[-1]
        if not test_phase:
            mask = torch.nn.functional.interpolate(self.mask.unsqueeze(1), size=self.tshape)[:, 0, :, :]
            mask = self.flatten(mask)
            mask = self.fc(mask)
            mask = self.unflatten(mask)
            mask = torch.stack([mask for _ in range(x.shape[1])], dim=1)
            x = x * mask
            return self.avgpool(x)
        else:
            return torch.nn.functional.avg_pool2d(x, x.shape[-2:])

class NewRes50(torch.nn.Module):

    def __init__(self, net, insize=256):
        super(NewRes50, self).__init__()
        self.net = net
        tshape = int(insize/32)
        self.net.avgpool = StructuredPool(tshape=(tshape, tshape))

    def forward(self, x):
        self.net.avgpool.build_mask(x)
        return self.net(x)

class CannyFilter(torch.nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        from canny_func import get_gaussian_kernel, get_sobel_kernel, get_thin_kernels
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = torch.nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = torch.nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = torch.nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)


        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = torch.nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = torch.nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges


class YOLOXood(torch.nn.Module):

    def __init__(self, yolox, postproc, input_side=128, yolox_thres=0.1, pretrained=True, ood_num_class=2):
        super(YOLOXood, self).__init__()
        self.yolox = yolox
        self.yolox_thres = yolox_thres
        self.input_side = input_side
        self.ood = torchvision.models.resnet50(pretrained=pretrained)
        if getattr(self.ood, "classifier", None) is not None:
            self.ood.classifier[-1] = torch.nn.Linear(
                in_features=self.ood.classifier[-1].in_features, 
                out_features=ood_num_class,
                bias=self.ood.classifier[-1].bias[0])
        elif getattr(self.ood, "fc", None) is not None:
            # fc_layers = torch.nn.Sequential(
            #     torch.nn.Linear(
            #     in_features=self.ood.fc.in_features, 
            #     out_features=self.ood.fc.in_features,
            #     bias=self.ood.fc.bias[0]),
            #     torch.nn.Dropout(p=0.2),
            #     torch.nn.Linear(
            #     in_features=self.ood.fc.in_features, 
            #     out_features=ood_num_class,
            #     bias=self.ood.fc.bias[0]),
            # )
            fc_layers = torch.nn.Linear(
                in_features=self.ood.fc.in_features, 
                out_features=ood_num_class,
                bias=self.ood.fc.bias[0])
            self.ood.fc = fc_layers
            # for p in self.ood.parameters():
            #     if p.shape[0] == ood_num_class: break
            #     p.requires_grad = False
        self.ood = NewRes50(self.ood, insize=input_side)
        self.postproc = postproc
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if self.yolox.training:
            self.yolox.eval()

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target.long())

    def forward(self, x, target=None, yolox_inference=False):
        if not yolox_inference and target != None:
            pred = self.ood(x)
            return {'total_loss': self.get_loss(pred, target)}
        elif not yolox_inference and target == None:
            pred = self.ood(x)
            return pred
        # b, c, w, h = x.shape
        # detections = self.yolox(x)
        # for bi in range(b):
        #     det = detections[bi]
        #     bboxes = det[:, 0:4]
        #     scores = det[:, 4] * det[:, 5]
        #     ids = torch.where(scores >= self.yolox_thres)[0]
        #     for i in ids:
        #         cx, cy, bw, bh = bboxes[i]
        #         x1, y1 = cx - self.input_side/2, cy - self.input_side/2
        #         x2, y2 = cx + self.input_side/2, cy + self.input_side/2
        #         x1, y1, x2, y2 = shift_box_in_image(x1, y1, x2, y2, w, h, self.input_side)
        #         # _, is_ood = self.ood(x[bi:bi+1, :, y1:y2, x1:x2]).max(dim=-1)
        #         ood_conf = self.ood(x[bi:bi+1, :, y1:y2, x1:x2]).softmax(dim=-1)[0, 1]
        #         detections[bi, i, 4] = detections[bi, i, 4] * ood_conf
        
        # if self.postproc:
        #     return [detections, torch.empty_like(detections)]
        # return detections


def get_box_ood_label(bboxes, t_boxes, iou_thres, input_side, tile_list=None, inp=None, bboxes_is_xywh=False):
    if not bboxes_is_xywh:
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
        bboxes[:, :2] = bboxes[:, :2] + bboxes[:, 2:]/2
    labels = []
    boxes = []
    iou_seen = []
    for i in range(bboxes.shape[0]):
        box = bboxes[i]
        iou = bboxes_iou(bboxes[i:i+1], t_boxes, xyxy=False)[0]
        bids = iou.argsort(descending=True)
        iou_flag = False
        for bid in bids:
            if iou[bid] >= iou_thres and bid not in iou_seen:
                iou_flag = True
                iou_seen.append(bid)
                break
        boxes.append(box)
        if tile_list != None:
            cx, cy, bw, bh = box
            x1, y1 = cx - bw/2, cy - bh/2
            x2, y2 = cx + bw/2, cy + bh/2
            x1 = max(0, x1.int())
            x2 = min(inp.shape[-2], x2.int())
            y1 = max(0, y1.int())
            y2 = min(inp.shape[-1], y2.int())
            # x1, y1 = cx - input_side/2, cy - input_side/2
            # x2, y2 = cx + input_side/2, cy + input_side/2
            # x1, y1, x2, y2 = shift_box_in_image(x1, y1, x2, y2, inp.shape[-2], inp.shape[-1], input_side)
            tile = inp[:, y1:y2, x1:x2].detach().cpu()
            tile_list.append(tile)
        if iou_flag:
            labels.append(torch.as_tensor(1))
        else:
            labels.append(torch.as_tensor(0))
    return labels, boxes, tile_list


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.input_size = (256, 256)  # OOD net input size
        self.test_size = (1024, 1024)  # YOLOX net input size
        self.test_conf = 0.1
        self.nmsthre = 0.5
        # ---------- YOLOX l ---------- #
        self.depth = 1.0
        self.width = 1.0
        # ---------- YOLOX l ---------- #
        self.warmup_epochs = 1
        self.iou_thres = 0.1
        self.resize_input = False

        # ---------- transform config ------------ #
        self.mosaic_prob = 0
        self.mixup_prob = 0
        self.hsv_prob = 0
        self.flip_prob = 0.5
        self.degrees = 0

        # ------------
        self.model = self.get_model(depthwise=True, postproc=False)
        self.save_trainimg = False
        self.model_has_head = True
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


    def get_model(self, postproc=False, **kwargs):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, **kwargs)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, **kwargs)
            yolox = YOLOX(backbone, head, False)
            ckpt = torch.load('./YOLOX_outputs/glandular_dw/best_ckpt.pth', map_location="cpu")['model']
            yolox.load_state_dict(ckpt)
            yolox.eval()
            self.model = YOLOXood(yolox, postproc, input_side=self.input_size[0], yolox_thres=self.test_conf)
            
        self.model.yolox.apply(init_yolo)
        self.model.yolox.head.initialize_biases(1e-2)
        return self.model

    def before_iter(self):
        self.model.yolox.eval()

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False, data_type=None):
        tag = '_onlybox' # ''
        odd_data_pre = "ODD_data%s_testconf%f_inputsize%d" % (tag, self.test_conf, self.input_size[0])
        manual_ood_data_pre = 'ODD_data_manual_nonsenseFP%s_inputsize%d' % (tag, self.input_size[0])
        if not os.path.exists("%s_bbox.pth" % odd_data_pre):
            self.model.yolox.eval()
            det_trainloader = self.get_orig_data_loader(batch_size, is_distributed, no_aug, cache_img)
            orig_img_list, bbox_list, label_list = None, [], []
            tile_list = []
            batch_n  = 0
            for inps, targets, _, _ in tqdm(det_trainloader, desc="Preparing data for OOD training"):
                batch_n += 1
                # if batch_n == 10: break
                inps = inps.to(data_type).cuda()
                targets = targets.to(data_type).cuda()
                targets.requires_grad = False
                with torch.no_grad():
                    detections = self.model.yolox(inps)
                
                detections = postprocess(
                    detections, self.num_classes, self.test_conf, nms_thre=0
                )

                for bi in range(len(detections)):
                    target = targets[bi]
                    target = target[torch.where(target.sum(dim=1) != 0)]
                    det = detections[bi]
                    if det is None: continue
                    labels, boxes, tile_list = get_box_ood_label(
                        det[:, 0:4], 
                        target[:, 1:], 
                        self.iou_thres, 
                        self.input_size[0], 
                        tile_list=tile_list, 
                        inp=inps[bi]
                    )
                    if len(labels) == 0: continue
                    labels = torch.stack(labels)
                    boxes = torch.stack(boxes)
                    label_list.append(labels.detach().cpu())
                    bbox_list.append(boxes.detach().cpu())

            torch.save(bbox_list, "%s_bbox.pth" % odd_data_pre)
            torch.save(label_list, "%s_label.pth" % odd_data_pre)
            torch.save(tile_list, "%s_tile.pth" % odd_data_pre)
        orig_img_list = None
        bbox_list = torch.load("%s_bbox.pth" % odd_data_pre)
        label_list = torch.load("%s_label.pth" % odd_data_pre)
        tile_list = torch.load("%s_tile.pth" % odd_data_pre)
        if os.path.exists("%s_tile.pth" % manual_ood_data_pre):
            manual_tile_list = torch.load("%s_tile.pth" % manual_ood_data_pre)
            idx = torch.where(torch.cat(label_list)==1)[0]
            tile_list = [tile_list[i] for i in idx] + manual_tile_list
            label_list = [torch.as_tensor([1]) for _ in idx]
            label_list += [torch.as_tensor([0]) for _ in range(len(manual_tile_list))]
            
        dataset = OODDataset(orig_img_list, bbox_list, label_list, self.input_size[0], tile_list=tile_list)
        self.dataset = dataset
        length = len(self.dataset)
        train_size, validate_size=int(0.8*length), length - int(0.8*length)
        self.train_set, self.validate_set=torch.utils.data.random_split(self.dataset, [train_size, validate_size])

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            # "sampler": sampler,
            "shuffle": True,
            "batch_size": batch_size,
            # "collate_fn": collate_fn
        }
        train_loader = torch.utils.data.DataLoader(self.train_set, **dataloader_kwargs)
        return train_loader

    def get_orig_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):

        from yolox.data import (
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )

        dataset = get_dataset("datasets/train_glandular.csv",
                              datatype="train",
                              transform=get_transform(train=True))

        self.dataset = dataset

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=False,
        )
        ##################################################################
        # dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        # dataloader_kwargs["batch_sampler"] = batch_sampler
        # # Make sure each process has different random seed, especially for 'fork' method
        # dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        # train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        # train_loader.close_mosaic()
        ##################################################################
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            # "sampler": sampler,
            "shuffle": True,
            "batch_size": batch_size,
            "collate_fn": collate_fn
        }
        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)
        ##################################################################
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):

        valdataset = get_dataset("datasets/val_glandular.csv",
                                  datatype="val",
                                  transform=get_transform(train=False))

        valdataset = self.validate_set
        valdataset.is_eval = True
        sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            # "sampler": sampler,
            "shuffle": False,
            "batch_size": 1
        }
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
    #
    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = glandular_evaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            ood_iou_thres=self.iou_thres,
            ood_input_side=self.input_size[0]
        )
        return evaluator

