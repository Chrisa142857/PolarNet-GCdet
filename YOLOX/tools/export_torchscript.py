#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch

from yolox.exp import get_exp
from yolox.utils import postprocess

def make_parser():
    parser = argparse.ArgumentParser("YOLOX torchscript deploy")
    parser.add_argument(
        "--output-name", type=str, default="detector_yolox_l_dw_input1x3x1024x1024.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/glandular_dw.py",
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model(postproc=True).cuda()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    model.postprocess = True
    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")
    print(model.postprocess)
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    # dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1]).cuda()
    side = 1024
    bsize = 36
    dummy_input = torch.randn(bsize, 3, side, side).cuda()
    # from PIL import Image
    # from torchvision.transforms import PILToTensor
    # example_input = Image.open(r'E:\wei\glandular1536\images\5886liuaie.mrxs_x4398_y79717_side1536.jpg');
    # example_input = PILToTensor()(example_input).unsqueeze(0)
    # example = torch.zeros(1, 3, side, side)
    # example[:, :, :1536, :1536] = example_input
    # example = example.cuda()
    with torch.no_grad():
        # print(model(dummy_input))
        # exit()
        mod = torch.jit.trace(model, dummy_input, check_trace=False)
    # mod.save(args.output_name)
    mod.save("traced_model/detector_yoloxpolar1+2_input36x3x1024x1024.pt")
    logger.info("generated torchscript model named {}".format(args.output_name))


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    main()
