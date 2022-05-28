#!/usr/bin/env python
# coding=utf-8
"""
run script
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import PIL
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import argparse
import sys
import os

from utils.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone
from faster_rcnn import FasterRCNN, FastRCNNPredictor
from faster_polar_rcnn import FasterPolarRCNN
from datasets import transforms as T
from torchvision import transforms
from datasets.data import get_dataset
import _utils
from tools import engine
from utils.darknet_models import Darknet
from get_flop import get_model_info

from polar_net import PolarNet

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
            T.Normalize() 
        ]
        return T.Compose(transform_list)
  
# def get_transform(train):
#     transforms_list = []
#     transforms_list.append(T.ToTensor())
#     if train:
#         transforms_list.append(T.RandomHorizontalFlip(0.5))
#         transforms_list.append(T.RandomVerticalFlip(0.5))

#     return T.Compose(transforms_list)


def parse_args(args):
    parser = argparse.ArgumentParser(description="TCT object detection")
    subparsers = parser.add_subparsers(
        help="optimizer type",
        dest="optimizer_type"
    )
    subparsers.required = True

    parser.add_argument("--use_yolco",
                        help="use yolco model",
                        type=int,
                        default=0)
    parser.add_argument("--model_name",
                        help="backbone",
                        type=str,
                        default="resnet50")
    parser.add_argument("--pretrained",
                        help="whether use pretrained weight",
                        type=bool,
                        default=True)
    parser.add_argument("--device",
                        help="cuda or cpu",
                        type=str,
                        default="cuda:3")
    parser.add_argument("--seed",
                        help="seed",
                        type=int,
                        default=7)
    parser.add_argument("--root",
                        help="image root dir",
                        type=str,
                        default="/home/stat-caolei/code/FasterDetection/data/VOC2007_/")
    parser.add_argument("--train_batch_size",
                        help="train batch size",
                        type=int,
                        default=2)
    parser.add_argument("--val_batch_size",
                        help="val batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--test_batch_size",
                        help="test batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--num_workers",
                        help="number of workers",
                        type=int,
                        default=12)
    parser.add_argument("--log_dir",
                        help="tensorboard log dir",
                        type=str,
                        default="./logs")
    sgd_parser = subparsers.add_parser("SGD")
    sgd_parser.add_argument("--sgd_lr",
                            help="SGD learning rate",
                            type=float,
                            default=0.005)
    sgd_parser.add_argument("--momentum",
                            help="SGD momentum",
                            type=float,
                            default=0.9)
    sgd_parser.add_argument("--weight_decay",
                            help="SGD weight decay",
                            type=float,
                            default=5e-4)
    adam_parser = subparsers.add_parser("Adam")
    adam_parser.add_argument("--adam_lr",
                             help="Adam learning rate",
                             type=float,
                             default=0.01)
    parser.add_argument("--step_size",
                        help="StepLR",
                        type=int,
                        default=8)
    parser.add_argument("--gamma",
                        help="StepLR gamma",
                        type=float,
                        default=0.1)
    parser.add_argument("--num_epochs",
                        help="number of Epoch",
                        type=int,
                        default=50)
    parser.add_argument("--save_model_path",
                       help="model saving dir",
                       type=str,
                       default="./results/saved_models/resnet50.pth")
    parser.add_argument("--record_iter",
                       help="record step",
                       type=int,
                       default=10)
    parser.add_argument("--voc_results_dir",
                        help="pred boxes dir",
                        type=str,
                        default="/home/stat-caolei/code/Final_TCT_Detection/tmp/detection_results/")
    parser.add_argument("--pretrained_resnet50_coco",
                        help="whether use resnet50 coco pretrained weight",
                        type=int,
                        default=0)
    parser.add_argument("--ReduceLROnPlateau",
                        help="lr decay method",
                        type=bool,
                        default=False)
    parser.add_argument("--start_epoch",
                        help="resume epochs",
                        type=int,
                        default=0)
    parser.add_argument("--resume",
                        help="resume training",
                        type=int,
                        default=0)
    parser.add_argument("--use_attn",
                        help="use attn fpn",
                        type=int,
                        default=0)
    parser.add_argument("--use_polar",
                        help="use polar net",
                        type=int,
                        default=0)
    parser.add_argument("--update_polar_only",
                        help="only update polar net",
                        type=int,
                        default=0)
    parser.add_argument("--polar_type",
                        help="old version, please ignore",
                        type=str,
                        default='v4')
    parser.add_argument("--use_new_v",
                        help="set 1 use polar attention to update feature map (QKV) | set 0 use initial feature map (QK)",
                        type=int,
                        default=0)
    parser.add_argument("--data_tail",
                        help="temp csv name tail",
                        type=str,
                        default="")

    return parser.parse_args(args)


def main(args=None):
    model_urls = {
        "fasterrcnn_resnet50_fpn_coco":
        "http://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    }
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    torch.cuda.set_device(int(args.device[-1]))
    device = torch.device('cuda')
    
    CLASSES = {"__background__", "nGEC", "AGC"}
    if args.use_yolco == 1:
        # backbone = denseInCnet_fpn_backbone(args.model_name, args.pretrained)
        backbone = Darknet()
        model = FasterRCNN(backbone, num_classes=len(CLASSES), min_size=(896, 960, 1024, 1088, 1152))

    elif args.pretrained_resnet50_coco == 1:
        backbone = resnet_fpn_backbone("resnet50", False, use_attn=args.use_attn)
        if args.use_polar == 1:
            # polar_net = [PolarNet().to(device) for _ in range(5)]
            polar_net = PolarNet(mtype=args.polar_type)
            model = FasterPolarRCNN(backbone, polar_net, mtype=args.polar_type, num_classes=len(CLASSES))
        else:
            model = FasterRCNN(backbone, num_classes=len(CLASSES))
        # state_dict = load_state_dict_from_url(
        #     model_urls["fasterrcnn_resnet50_fpn_coco"],
        #     progress=True
        # )
        # model.load_state_dict(state_dict)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          len(CLASSES))
        
    else:
        # backbone = resnet_fpn_backbone(args.model_name, args.pretrained)
        # model = FasterRCNN(backbone, num_classes=len(CLASSES))
        backbone = densenet_fpn_backbone(args.model_name, args.pretrained)
        model = FasterRCNN(backbone, num_classes=len(CLASSES), min_size=(896, 960, 1024, 1088, 1152))

    if args.resume != 0:
        model.load_state_dict(torch.load(args.save_model_path.replace('best', 'last')))
    print(model)
    print(get_model_info(model, (1024,1024)))


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load data
    print("====Loading data====")
    # dataset = get_dataset("./statistic_description/tmp/train.csv",
    #                       datatype="train",
    #                       transform=get_transform(train=True))
    # dataset_val = get_dataset("./statistic_description/tmp/val.csv",
    #                           datatype="val",
    #                           transform=get_transform(train=False))
    # dataset_test = get_dataset("./statistic_description/tmp/test.csv",
    #                            datatype="test",
    #                            transform=get_transform(train=False))
    dataset = get_dataset("train%s.csv"%args.data_tail,
                          datatype="train",
                          transform=get_transform(train=True))
    dataset_val = get_dataset("val%s.csv"%args.data_tail,
                              datatype="val",
                              transform=get_transform(train=False))
    # dataset_test = get_dataset("./statistic_description/tmp/xiugao_test.csv",
    #                            datatype="test",
    #                            transform=get_transform(train=False))
    # dataset = get_custom_voc(
    #     root=args.root,
    #     transforms=get_transform(train=True),
    #     dataset_flag="train"
    # )
    # dataset_val = get_custom_voc(
    #     root=args.root,
    #     transforms=get_transform(train=False),
    #     dataset_flag="val"
    # )
    # dataset_test = get_custom_voc(
    #     root=args.root,
    #     transforms=get_transform(train=False),
    #     dataset_flag="test"
    # )

    print("====Creating dataloader====")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_utils.collate_fn
    )
    # dataloader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.test_batch_size, shuffle=False,
    #     num_workers=8,
    #     collate_fn=_utils.collate_fn
    # )
    dataloaders = {
        "train": dataloader,
        "val": dataloader_val,
        # "test": dataloader_test,
    }
    logdir = args.log_dir
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    print("====Loading model====")
    model.to(device)
    if args.update_polar_only == 1:
        for n, p in model.named_parameters():
            if "polar_net" not in n:
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD([
        #     {"params": [p for p in model.backbone.body.parameters()
        #                 if p.requires_grad]},
        #     {"params": [p for p in model.backbone.fpn.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        #     {"params": [p for p in model.backbone.bottom_up.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        #     {"params": [p for p in model.rpn.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        #     {"params": [p for p in model.roi_heads.parameters()
        #                 if p.requires_grad], "lr": 1e-2},
        # ], lr=1e-3, momentum=args.momentum,
        # weight_decay=args.weight_decay)

        if args.ReduceLROnPlateau:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.1, patience=10,
                verbose=False, threshold=0.0001, threshold_mode="rel",
                cooldown=0, min_lr=0, eps=1e-8
            )
        else:
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer, step_size=args.step_size, gamma=args.gamma
            # )
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[8, 24], gamma=args.gamma
            # )
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[25, 50, 80], gamma=args.gamma
            )
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[50, 60], gamma=args.gamma
            # )

    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.adam_lr)
        lr_scheduler = None

    print("====Start training====")
    # engine.train_process(model=model, optimizer=optimizer,
    #                      lr_sche=lr_scheduler,
    #                      dataloaders=dataloaders,
    #                      num_epochs=args.num_epochs,
    #                      use_tensorboard=True,
    #                      device=device,
    #                      save_model_path=args.save_model_path,
    #                      record_iter=args.record_iter,
    #                      writer=writer,
    #                      ReduceLROnPlateau=args.ReduceLROnPlateau)
    if args.use_yolco:
        model_name = 'icn'
    else:
        model_name = args.model_name
    engine.train_process(model=model, optimizer=optimizer,
                         lr_sche=lr_scheduler,
                         dataloaders=dataloaders,
                         num_epochs=args.num_epochs,
                         use_tensorboard=True,
                         device=device,
                         save_model_path=args.save_model_path,
                         record_iter=args.record_iter,
                         writer=writer,
                         ReduceLROnPlateau=args.ReduceLROnPlateau, start_epoch=args.start_epoch,
                         val_csvname='val%s.csv'%args.data_tail,
                         model_name=model_name, update_polar_only=args.update_polar_only)


if __name__ == "__main__":
    main()

