#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw, ImageFont
from datasets.transforms import denormalize
from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def save_trainimg(self, image, target=None, pred=None):
        font = ImageFont.truetype(r'assets/arial.ttf', 40)
        # if pred == None:
        save_img = ToPILImage()(denormalize(image.detach().cpu()))
        # else:
        #     save_img = ToPILImage()(image)
        if target != None and len(target.shape) > 1:
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
                        box_xyxy[k].tolist()[:2], "class%d:%.2f" % (label[k].item(), score[k].item()),
                        font=font, fill='red'
                    )
                    
        if target != None and len(target.shape) > 1:
            for j in range(true_box_xyxy.shape[0]):
                color = (255, 215, 0)
                draw.rectangle(
                    true_box_xyxy[j].tolist(), outline=color,
                    width=5
                )
                draw.text(
                    true_box_xyxy[j].tolist(), "class%d" % true_label[j].item(),
                    font=font, fill='red'
                )
        elif len(target.shape) == 1:
            draw.text([10,10], "class: %d" % target.item(), 
                    font=font, fill='red'
            )
            
        if pred == None:
            name = 'train_batch%d.jpg' % (self.iter + 1)
        else:
            name = 'val_pred'
        save_img = ToTensor()(save_img)
        self.tblogger.add_image(name, save_img,
                         global_step=self.epoch)

    def train_in_iter(self):
        # return
        for self.iter, (inps, targets, _, _) in enumerate(self.train_loader):
            # if (self.iter + 1) % 100 == 0:
            # if self.exp.save_trainimg:
            # self.save_trainimg(inps[0], targets[0])
            # else:
            #     self.save_trainimg(inps[0], targets[0])
            self.before_iter()

            iter_start_time = time.time()
            # inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type).cuda()
            targets = targets.to(self.data_type).cuda()
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets)

            loss = outputs["total_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            iter_end_time = time.time()
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs,
            )

            self.after_iter()

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        # self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.no_aug = True
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
            data_type=self.data_type
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            # self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.exp.model_has_head:
                if self.is_distributed:
                    if getattr(self.model, "yolox", None) is None:
                        self.model.module.head.use_l1 = True
                    else:
                        self.model.yolox.module.head.use_l1 = True
                else:
                    if getattr(self.model, "yolox", None) is None:
                        self.model.head.use_l1 = True
                    else:
                        self.model.yolox.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

        if getattr(self.train_loader.dataset, "shuffle_idx", None) is not None:
            self.train_loader.dataset.shuffle_idx()

    def after_epoch(self):
        self.save_ckpt(ckpt_name="epoch%d"%self.epoch)
        self.save_ckpt(ckpt_name="lastest")
        self.save_ckpt(ckpt_name="last_epoch")

        # if (self.epoch + 1) % self.exp.eval_interval == 0:
        #     all_reduce_norm(self.model)
        #     self.evaluate_and_save_model()

    def before_iter(self):
        if getattr(self.exp, "before_iter", None) is not None:
            self.exp.before_iter()
        

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            for k, v in loss_meter.items():
                self.tblogger.add_scalar(k, v.latest, self.epoch * len(self.train_loader) + self.iter + 1)
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0 and self.exp.resize_input:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap_dict, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            ap_info = ''
            for key in ap_dict:
                if 'val_save' in key: continue
                self.tblogger.add_scalar(key, ap_dict[key], self.epoch + 1)
                ap_info += "%s: %.3f; " % (key, ap_dict[key])
            logger.info("\n" + ap_info + "\n" + summary)
        synchronize()
        # mAP, ap_n = 0, 0
        # for k in ap_dict:
        #     if 'all' in k and 'ap' in k:
        #         mAP += ap_dict[k]
        #         ap_n += 1
        # mAP /= ap_n
        mAP = ap_dict['all_50_ap']
        self.save_ckpt("last_epoch", mAP > self.best_ap)
        self.best_ap = max(self.best_ap, mAP)
        # if 'val_save_img' in ap_dict:
        #     self.save_trainimg(
        #         ap_dict['val_save_img'],
        #         ap_dict['val_save_label'],
        #         [
        #             ap_dict['val_save_pred_box'],
        #             ap_dict['val_save_pred_cls'],
        #             ap_dict['val_save_pred_score']
        #         ]
        #     )

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
