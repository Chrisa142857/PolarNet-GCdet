# encoding: utf-8
import os

import torch
import torch.distributed as dist
from torchvision import transforms
import time

import pandas as pd
import numpy as np
from collections import namedtuple

from yolox.exp import Exp as MyExp
from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized
from datasets import transforms as T
from datasets.data import get_dataset, is_box_valid_after_crop, collate_fn

from tqdm import tqdm

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
                # if not is_box_valid_after_crop(crop_x, crop_y, x1, x2, y1, y2, new_side):
                #     continue
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
    ood_fp_num = 0

    # loop over each pred box to see if it matches one ground box
    if nd > 0:
        for d in range(nd):
            # one pred box coords
            bb = preds[d].coordinates.astype(float)
            pred_type = preds[d].object_type
            image_path = preds[d].image_path
            if preds[d].image_path in object_dict:
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
                ood_fp_num += 1
                fp[d] = 1.
                fp_nonclass[d] = 1.
                fp_subclass[int(pred_type)][d] = 1.
    print("ood_fp_num", ood_fp_num)
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
    recs['all_%d_recall'%(ovthresh*100)] = rec
    precs['all_%d_prec'%(ovthresh*100)] = prec
    aps['all_%d_ap'%(ovthresh*100)] = ap
    rec = tp_nonclass / float(num_object)
    prec = tp_nonclass / np.maximum(tp_nonclass + fp_nonclass, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    recs['nonclass_%d_recall'%(ovthresh*100)] = rec
    precs['nonclass_%d_prec'%(ovthresh*100)] = prec
    aps['nonclass_%d_ap'%(ovthresh*100)] = ap

    for i, (tp, fp, num_object) in enumerate(zip(tp_subclass, fp_subclass, num_object_subclass)):
        rec = tp / float(num_object)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        recs['class%d_%d_recall'%(i, ovthresh*100)] = rec
        precs['class%d_%d_prec'%(i, ovthresh*100)] = prec
        aps['class%d_%d_ap'%(i, ovthresh*100)] = ap


    return recs, precs, aps


class glandular_evaluator:

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
        gt_csv_path,
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
        model.eval()

        preds = []
        locs = []
        rand_iter_id = np.random.randint(0, n_samples)
        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            imgs = imgs.type(torch.cuda.FloatTensor)

            is_time_record = cur_iter < len(self.dataloader) - 1

            if is_time_record:
                start = time.time()
            outputs = model(imgs)

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
                bboxes = []
                cls = []
                scores = []
            else:
                output = outputs[-1].cpu()
                bboxes = output[:, 0:4]
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]

            if cur_iter == rand_iter_id:
                val_save_img = imgs[0].detach().cpu()
                val_save_label = targets[0].detach().cpu()
                val_save_pred_box = bboxes
                val_save_pred_cls = cls
                val_save_pred_score = scores

            if outputs[-1] == None:
                continue
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
        recall, precision, ap = custom_voc_eval(self.gt_csv_path, self.loc_csv_path, ovthresh=0.5, num_class=2)
        # aps = [recall, precision, ap]
        aps = [ap]
        result = {k: ap[k] for ap in aps for k in ap}

        # acc = ((pred >= 0.5) == gt_label).mean()
        # fpr, tpr, _ = roc_curve(gt_label, pred)
        # roc_auc = auc(fpr, tpr)
        # return ap, acc, roc_auc
        result['val_save_img'] = val_save_img
        result['val_save_label'] = val_save_label
        result['val_save_pred_box'] = val_save_pred_box
        result['val_save_pred_cls'] = val_save_pred_cls
        result['val_save_pred_score'] = val_save_pred_score
        return result, info


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.input_size = (1024, 1024)
        self.test_size = (1024, 1024)
        self.test_conf = 0.1
        self.nmsthre = 0.5
        # ---------- YOLOX l ---------- #
        self.depth = 1.0
        self.width = 1.0
        # ---------- YOLOX l ---------- #
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 0
        self.mixup_prob = 0
        self.hsv_prob = 0
        self.flip_prob = 0.5
        self.degrees = 0

        # ------------
        self.model = self.get_model(depthwise=True, postproc=False)
        self.save_trainimg = False
        self.resize_input = True
        self.model_has_head = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 100

    def get_model(self, postproc=False, **kwargs):
        from yolco_model import Darknet

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            self.model = Darknet('yolco_cfg.cfg', lite_mode=True)
            
        self.model.apply(init_yolo)
        return self.model


    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False, data_type=None):

        from yolox.data import (
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )

        dataset = get_dataset("datasets/glandular1536_20211217/8-1-1/train_glandular.csv",
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

    def get_eval_loader(self, batch_size, is_distributed, gt_csv_path, testdev=False, legacy=False):

        valdataset = get_dataset(gt_csv_path,
                                datatype="val",
                                transform=get_transform(train=False))


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
        if testdev:
            gt_csv_path='datasets/glandular1536_20211217/8-1-1/test_glandular_w_ood.csv'
        else:
            gt_csv_path='datasets/glandular1536_20211217/8-1-1/val_glandular.csv'
        val_loader = self.get_eval_loader(batch_size, is_distributed, gt_csv_path, testdev, legacy)

        evaluator = glandular_evaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            # confthre=self.test_conf,
            confthre=0.5,
            nmsthre=self.nmsthre,
            # nmsthre=0.1,
            num_classes=self.num_classes,
            gt_csv_path=gt_csv_path
        )
        return evaluator

