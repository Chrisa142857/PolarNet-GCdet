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

from sklearn.metrics import confusion_matrix
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

def save_fg_net_trainimg(iter, model, image, tgt, pred=None):
    target = tgt[torch.where(tgt.sum(dim=1) != 0)]
    nlabel = target.shape[0]
    true_box_xywh = target[:, 1:]
    true_box_xyxy = torch.zeros_like(true_box_xywh)
    true_box_xyxy[:, :2] = true_box_xywh[:, :2] - true_box_xywh[:, 2:] / 2
    true_box_xyxy[:, 2:] = true_box_xywh[:, :2] + true_box_xywh[:, 2:] / 2
    fg_label = model.squarer.square_target(true_box_xywh, nlabel, image.shape[-2])
    ratio = model.squarer.input_size / image.shape[-2]
    w = model.squarer.square_size
    h = model.squarer.square_size
    xywh = []
    for label, grid in zip(fg_label, model.squarer.grids):
        x1, y1 = grid
        x = x1 + model.squarer.square_size/2
        y = y1 + model.squarer.square_size/2
        x, y = x/ratio, y/ratio
        w = model.squarer.square_size/ratio
        h = model.squarer.square_size/ratio
        xywh.append([label, x, y, w, h])

    gt = torch.cat([torch.Tensor(xywh)], dim=0)
    if pred != None:
        xyxy = []
        labels = []
        scores = []
        for out, grid in zip(pred, model.squarer.grids):
            c, l = torch.softmax(out, dim=-1).max(dim=-1)
            x1, y1 = grid
            x2 = x1 + model.squarer.square_size
            y2 = y1 + model.squarer.square_size
            xyxy.append([x1/ratio, y1/ratio, x2/ratio, y2/ratio])
            labels.append([l.item()])
            scores.append([c.item()])
        pred = [torch.Tensor(xyxy), torch.Tensor(labels), torch.Tensor(scores)]
    image = save_trainimg(iter, image, gt, pred, prefix='fg=', log_name='fgnet', denorm=True, save=False)
    save_trainimg(iter, image, tgt, pred=None, prefix='class=', log_name='fgnet', denorm=False, save=True)

def save_trainimg(iter, image, target, pred=None, prefix='class', log_name='', denorm=True, save=True):
    from PIL import ImageDraw, ImageFont
    from torchvision.transforms import ToPILImage, ToTensor
    from datasets.transforms import denormalize

    font = ImageFont.truetype(r'assets/arial.ttf', 40)
    font1 = ImageFont.truetype(r'assets/arial.ttf', 30)
    if denorm:
        image = denormalize(image.detach().cpu())
    else:
        image = image.detach().cpu()
    save_img = ToPILImage()(image)
    target = target[torch.where(target.sum(dim=1) != 0)]
    true_box_xywh = target[:, 1:]
    true_box_xyxy = torch.zeros_like(true_box_xywh)
    true_box_xyxy[:, :2] = true_box_xywh[:, :2] - true_box_xywh[:, 2:]/2
    true_box_xyxy[:, 2:] = true_box_xywh[:, :2] + true_box_xywh[:, 2:]/2
    true_label = target[:, :1]
    draw = ImageDraw.Draw(save_img)
    if pred != None:
        box_xyxy, label, score = pred
        for k in range(box_xyxy.shape[0]):
            if score[k].item() > 0.01:
                color = (0, 255, 0)
                draw.rectangle(box_xyxy[k].tolist(),
                                outline=color, width=5)
                draw.text(
                    box_xyxy[k].tolist(), "%s%d:%.2f" % (prefix, label[k].item(), score[k].item()),
                    font=font1, fill='green'
                )
    for j in range(true_box_xyxy.shape[0]):
        color = (255, 215, 0)
        draw.rectangle(
            true_box_xyxy[j].tolist(), outline=color,
            width=4
        )
        draw.text(
            true_box_xyxy[j].tolist(), "%s%d" % (prefix, true_label[j].item()),
            font=font, fill='red'
        )
    if pred == None:
        name = 'train%s_batch%d.jpg' % (log_name, iter + 1)
    else:
        name = 'val%s_iter%d_pred.jpg' % (log_name, iter + 1)
    if save:
        save_img.save(os.path.join('temp', name))
    else:
        return ToTensor()(save_img)

class glandular_evaluator:

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
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
        self.best_fg_acc = 0.0

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
        save_out=True
    ):
        preds = []
        labels = []
        fg_acc = 0.0
        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        progress_bar = tqdm
        old_side = 1536
        new_side = 1024
        crop_x = int(old_side / 2) - int(new_side / 2)
        crop_y = int(old_side / 2) - int(new_side / 2)
        record_iter = np.random.randint(0, 100)
        model.eval()

        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            imgs = imgs.type(torch.cuda.FloatTensor)

            is_time_record = cur_iter < len(self.dataloader) - 1

            if is_time_record:
                start = time.time()
            fg_outputs, fg_scores = model(imgs)
            
            if save_out:
                for i in range(len(targets)):
                    if targets[i].shape[0] == 0: 
                        continue
                    else:
                        save_fg_net_trainimg(cur_iter, model, imgs[i], targets[i], fg_outputs[i])
                        break


            acc, label, pred = model.get_batch_acc(fg_outputs, targets, imgs.shape[-2])
            fg_acc += acc
            preds.append(pred)
            labels.append(label)

            if is_time_record:
                infer_end = time_synchronized()
                inference_time += infer_end - start

        statistics = torch.cuda.FloatTensor([inference_time, n_samples])
        inference_time = statistics[0].item()
        n_samples = statistics[1].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "inference"],
                    [a_infer_time, a_infer_time],
                )
            ]
        )

        info = time_info + "\n"
        result = {}
        result['fg_acc'] = fg_acc / (n_samples * self.dataloader.batch_size)
        m = confusion_matrix(torch.cat(labels), torch.cat(preds))
        result['confusion_matrix[0, 0]'] = m[0][0]
        result['confusion_matrix[0, 1]'] = m[0][1]
        result['confusion_matrix[1, 0]'] = m[1][0]
        result['confusion_matrix[1, 1]'] = m[1][1]
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
        self.basic_lr_per_img = 5e-4 * 9 / 512

        # ---------- transform config ------------ #
        self.mosaic_prob = 0
        self.mixup_prob = 0
        self.hsv_prob = 0
        self.flip_prob = 0.5
        self.degrees = 0

        # ------------
        self.max_epoch = 40

        self.model = self.get_model(depthwise=True, postproc=False)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


    def get_model(self, postproc=False, **kwargs):
        from fg_networks import ForegroundNet

        if getattr(self, "model", None) is None:
            self.model = ForegroundNet(name="B_16", pretrained=True, input_size=672)

        return self.model


    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr
            )
            self.optimizer = optimizer

        return self.optimizer


    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):

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
        )
        return evaluator

