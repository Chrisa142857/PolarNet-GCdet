import math
import sys
import time
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import ImageDraw, ImageFont
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

import torchvision.models.detection.mask_rcnn
from torchvision.ops import nms

import sys
sys.path.append(".")
sys.path.append("..")
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils
from .voc_eval import write_custom_voc_results_file, do_python_eval
from .voc_eval_new import custom_voc_eval
# 使用tensorboard可视化损失
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
from tqdm import tqdm
import os
import copy


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[0] = x[0] * std[0] + mean[0]
    x[1] = x[1] * std[1] + mean[1]
    x[2] = x[2] * std[2] + mean[2]
    return x


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def train_process(model, optimizer, lr_sche,
                  dataloaders, num_epochs,
                  use_tensorboard,
                  device,
                 # model save params
                 save_model_path,
                 record_iter,
                 # tensorboard
                 imgsave_iter=100,
                 writer=None,
                 ReduceLROnPlateau=False,
                 start_epoch=0,
                 val_csvname='',
                 model_name='',
                 update_polar_only=0):
    font = ImageFont.truetype('arial.ttf', 40) 
    savefig_flag = True
    model.train()
    model.apply(freeze_bn)
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_polarity = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    best_score = 0.0
    best_stat_dict = copy.deepcopy(model.state_dict())
    for lr_epoch in range(start_epoch):
        if not ReduceLROnPlateau:
            lr_sche.step()
        else:
            val_mAP = custom_voc_evaluate(
                model, dataloaders["val"], device=device,
                gt_csv_path=val_csvname,
                cls_csv_path="temp/%s_epoch%d_cls.csv" % (model_name, epoch),
                loc_csv_path="temp/%s_epoch%d_loc.csv" % (model_name, epoch)
            )
            lr_sche.step(val_mAP)
    for epoch in range(start_epoch, num_epochs):
        lr_scheduler = None
        print("====Epoch {0}====".format(epoch))
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(dataloaders['train']) - 1)
            lr_scheduler = utils.warmup_lr_scheduler(optimizer,
                                                    warmup_iters,
                                                    warmup_factor)
        for i, (images, targets) in enumerate(tqdm(dataloaders['train']), 0):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                      for t in targets]

            if (i+1) % imgsave_iter == 0:
                save_img = ToPILImage()(denormalize(images[0].detach().cpu()))
                true_box = targets[0]['boxes']
                true_label = targets[0]['labels']
                draw = ImageDraw.Draw(save_img)
                for j in range(true_box.shape[0]):
                    if true_label[j].item() >= 0:
                        color = (255, 215, 0)
                        draw.rectangle(
                            true_box[j].tolist(), outline=color,
                            width=5
                        )
                        draw.text(
                            true_box[j].tolist(), "class%d"%true_label[j].item(),
                            font=font, fill='red'
                        )
                # save_img.save('logs/train_batch%d.jpg'%(i+1))

                save_img = ToTensor()(save_img)
                writer.add_image('train_batch%d.jpg'%(i+1), save_img,
                                global_step=epoch)


            optimizer.zero_grad()
            # 得到损失值字典
            loss_dict = model(images, targets)
            if update_polar_only == 1:
                losses = loss_dict['loss_polarity']
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict({'loss_polarity': loss_dict['loss_polarity']})
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            else:
                losses = sum(loss for loss in loss_dict.values())
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())


            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                optimizer.zero_grad()
                continue
                # sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            losses_total = losses.item()
            # roi分类损失
            loss_classifier = loss_dict['loss_classifier'].item()
            # roi回归损失
            loss_box_reg = loss_dict['loss_box_reg'].item()
            # rpn分类损失
            loss_objectness = loss_dict['loss_objectness'].item()
            # rpn回归损失
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()
            if 'loss_polarity' in loss_dict:
                loss_polarity = loss_dict['loss_polarity'].item()
            # 学习率
            lr = optimizer.param_groups[0]['lr']
            # lr_small = optimizer.param_groups[0]["lr"]
            # lr_large = optimizer.param_groups[1]["lr"]
            running_loss += losses_total
            running_loss_classifier += loss_classifier
            running_loss_box_reg += loss_box_reg
            running_loss_objectness += loss_objectness
            running_loss_rpn_box_reg += loss_rpn_box_reg
            if 'loss_polarity' in loss_dict:
                running_loss_polarity += loss_polarity

            if (i+1) % record_iter == 0:
                print('''Epoch{0} loss:{1:.4f}
                         loss_classifier:{2:.4f} loss_box_reg:{3:.4f}
                         loss_objectness:{4:.4f} loss_rpn_box_reg:{5:.4f} loss_polarity:{6:.4f}\n'''.format(
                          epoch,
                          losses_total, loss_classifier,
                          loss_box_reg, loss_objectness,
                          loss_rpn_box_reg, loss_polarity
                      ))
                if use_tensorboard:
                    # 写入tensorboard
                    writer.add_scalar("1. Total loss",
                                     running_loss / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("2. RoI classification loss",
                                     running_loss_classifier / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("2. RoI reg loss",
                                     running_loss_box_reg / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("2. RoI polarity loss",
                                     running_loss_polarity / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("3. RPN classification loss",
                                     running_loss_objectness / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("3. RPN reg loss",
                                     running_loss_rpn_box_reg / record_iter,
                                     epoch * len(dataloaders['train']) + i)
                    writer.add_scalar("lr",
                                     lr,
                                     epoch * len(dataloaders['train']) + i)
                    # writer.add_scalar("lr_small",
                    #                  lr_small,
                    #                  epoch * len(dataloaders['train']) + i)
                    # writer.add_scalar("lr_large",
                    #                  lr_large,
                    #                  epoch * len(dataloaders['train']) + i)
                    running_loss = 0.0
                    running_loss_classifier = 0.0
                    running_loss_box_reg = 0.0
                    running_loss_objectness = 0.0
                    running_loss_rpn_box_reg = 0.0
                    running_loss_polarity = 0.0

        # val_mAP, val_fig = custom_voc_evaluate(
        #     model, dataloaders['val'], device=device,
        #     voc_results_dir=voc_results_dir,
        #     savefig_flag=savefig_flag
        # )
        # val_mAP, acc, roc_auc = custom_voc_evaluate(
        #     model, dataloaders["val"], device=device,
        #     gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/val.csv",
        #     cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/cls.csv",
        #     loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/loc.csv"
        # )
        val_mAP = custom_voc_evaluate(
            model, dataloaders["val"], device=device,
            gt_csv_path=val_csvname,
            cls_csv_path="temp/%s_epoch%d_cls.csv" % (model_name, epoch),
            loc_csv_path="temp/%s_epoch%d_loc.csv" % (model_name, epoch)
        )
        # print("Epoch: ", epoch, "| val mAP: %.4f" % val_mAP,
        #       "| val acc: %.4f" % acc, "| val auc: %.4f" % roc_auc)
        # mmAP, ap_n = 0, 0
        # for k in val_mAP:
        #     if 'all' in k:
        #         mmAP += val_mAP[k]
        #         ap_n += 1
        # mmAP /= ap_n
        torch.save(copy.deepcopy(model.state_dict()), save_model_path.replace('best', 'last'))
        mmAP = val_mAP['all_50_ap']
        print("Epoch: ", epoch, "| val mAP50: %.4f" % mmAP)
        if not ReduceLROnPlateau:
            lr_sche.step()
        else:
            lr_sche.step(mmAP)
        if mmAP >= best_score:
            best_score = mmAP
            best_stat_dict = copy.deepcopy(model.state_dict())
            torch.save(best_stat_dict, save_model_path)
            savefig_flag = True
            for val_imgs, _, val_labels in dataloaders['val']:
                if np.random.randint(1,100) <= 5: break
            val_imgs = val_imgs[0].to(device)
            prediction = model([val_imgs])
            box = prediction[0]['boxes']
            score = prediction[0]['scores']
            label = prediction[0]['labels']
            true_box = val_labels[0]['boxes']
            true_label = val_labels[0]['labels']
            topil = ToPILImage()
            val_imgs = topil(denormalize(val_imgs.cpu()))
            draw = ImageDraw.Draw(val_imgs)
            for k in range(box.shape[0]):
                if label[k].item() >= 1 and score[k].item() > 0.5:
                    color = (0, 255, 0)
                    draw.rectangle(box[k].tolist(),
                                  outline=color, width=5)
                    draw.text(
                        box[k].tolist()[:2], "class%d:%.2f"%(label[k].item(), score[k].item()),
                        font=font, fill='red'
                    )
            for j in range(true_box.shape[0]):
                if true_label[j].item() >= 0:
                    color = (255, 215, 0)
                    draw.rectangle(
                        true_box[j].tolist(), outline=color,
                        width=5
                    )
                    draw.text(
                        true_box[j].tolist(), "class%d"%true_label[j].item(),
                        font=font, fill='red'
                    )
            val_imgs = ToTensor()(val_imgs)
            writer.add_image('val_pred', val_imgs,
                            global_step=epoch)
        else:
            savefig_flag = False
        if use_tensorboard:
            # writer.add_figure(
            #     "Validation PR-curve",
            #     val_fig,
            #     global_step=epoch
            # )
            for name in val_mAP:
                writer.add_scalar(
                    '0. Validation mAP %s' % name,
                    val_mAP[name],
                    global_step=epoch
                )
            # writer.add_scalar(
            #     "Validation acc",
            #     acc,
            #     global_step=epoch
            # )
            # writer.add_scalar(
            #     "Validation auc",
            #     roc_auc,
            #     global_step=epoch
            # )
        model.train()
        model.apply(freeze_bn)
        # print("Best Valid mAP: %.4f" % best_score)

    # print("====开始测试====")
    # model.load_state_dict(best_stat_dict)
    # # test_mAP, test_fig = custom_voc_evaluate(
    # #     model, dataloaders['test'],
    # #     device=device,
    # #     voc_results_dir=voc_results_dir,
    # #     savefig_flag=True
    # # )
    # # test_mAP, test_acc, test_roc_auc = custom_voc_evaluate(
    # #     model, dataloaders["test"], device=device,
    # #     gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/test.csv",
    # #     cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/cls.csv",
    # #     loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/loc.csv"
    # # )
    # test_mAP = custom_voc_evaluate(
    #     model, dataloaders["test"], device=device,
    #     gt_csv_path="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/xiugao_test.csv",
    #     cls_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/xiugao_cls.csv",
    #     loc_csv_path="/home/stat-caolei/code/TCT_V3/tmp/detection_results/xiugao_loc.csv"
    # )
    # # print("Test mAP: %.4f" % test_mAP,
    # #       "Test acc: %.4f" % test_acc,
    # #       "Test auc: %.4f" % test_roc_auc)
    # print("Test mAP: %.4f" % test_mAP)
    if use_tensorboard:
        # writer.add_figure(
        #     'Test PR-curve',
        #     test_fig,
        #     global_step=0
        # )
        writer.close()



# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types


# @torch.no_grad()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for image, targets in metric_logger.log_every(data_loader, 100, header):
#         image = list(img.to(device) for img in image)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(image)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator


@torch.no_grad()
def custom_voc_evaluate(model, data_loader, device,
                        gt_csv_path,
                        cls_csv_path,
                        loc_csv_path,
                        savefig_flag=False,
                        score_thres=0.1):
    old_side = 1536
    new_side = 1024
    crop_x = int(old_side/2) - int(new_side/2)
    crop_y = int(old_side/2) - int(new_side/2)
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'

    preds = []
    locs = []
    p_scores = []
    for image, label, _ in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                  for t in outputs]
        coords = []  # used to save pred coords x1 y1 x2 y2
        coords_score = []  # used to save pred box scores
        coords_label = []  # used to save pred box label
        if len(outputs[-1]["boxes"]) == 0:
            # if no pred boxes, means that the image is negative
            preds.append(0)
            coords.append([])
            coords_score.append("")
            locs.append("")
            p_scores.append(torch.empty(0))

        else:
            # preds.append(torch.max(outputs[-1]["scores"]).tolist())

            # we keep those pred boxes whose score is more than 0.1
            new_output_index = torch.where(outputs[-1]["scores"] >= score_thres)
            new_boxes = outputs[-1]["boxes"][new_output_index]
            new_scores = outputs[-1]["scores"][new_output_index]
            new_labels = outputs[-1]["labels"][new_output_index]
            p_scores.append(outputs[-1]["p_score"][new_output_index])
            if len(new_boxes) != 0:
                preds.append(torch.max(new_scores).tolist())
            else:
                preds.append(0)
            
            for i in range(len(new_boxes)):
                new_box = new_boxes[i].tolist()
                coords.append([new_box[0], new_box[1],
                               new_box[2], new_box[3]])
            coords_score += new_scores.tolist()
            coords_label += new_labels.tolist()
            line = ""
            for i in range(len(new_boxes)):
                if i == len(new_boxes) - 1:
                    line += str(coords_score[i]) + ' ' + str(int(coords_label[i]-1)) + ' ' + str(coords[i][0]+crop_x) + ' ' + \
                            str(coords[i][1]+crop_y) + ' ' + str(coords[i][2]+crop_x) + ' ' + \
                            str(coords[i][3]+crop_y)
                else:
                    line += str(coords_score[i]) + ' ' + str(int(coords_label[i]-1)) + ' ' + str(coords[i][0]+crop_x) + ' ' + \
                            str(coords[i][1]+crop_y) + ' ' + str(coords[i][2]+crop_x) + ' ' + \
                            str(coords[i][3]+crop_y) + ';'

            locs.append(line)
            
    torch.save({'p_score': p_scores}, cls_csv_path.replace('_cls', '_pscore').replace('.csv', '.pth'))
    print(len(data_loader.dataset.image_files_list), len(preds))
    cls_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_files_list,
         "prediction": preds}
    )
    print("====write cls pred results to csv====")
    cls_res.to_csv(cls_csv_path, columns=["image_path", "prediction"],
                   sep=',', index=None)
    loc_res = pd.DataFrame(
        {"image_path": data_loader.dataset.image_files_list,
         "prediction": locs}
    )
    print("====write loc pred results to csv====")
    loc_res.to_csv(loc_csv_path, columns=["image_path", "prediction"],
                   sep=',', index=None)
    # gt_anno = pd.read_csv(gt_csv_path, na_filter=False)
    # gt_label = gt_anno.annotation.astype(bool).astype(float).values
    # pred = cls_res.prediction.values
    # try:

    # aps = []
    # for i in range(1,10):
    recall, precision, ap = custom_voc_eval(gt_csv_path, loc_csv_path, ovthresh=0.5)
    aps = [ap] 
    result = {k: ap[k] for ap in aps for k in ap}
    # except:
    #     import ipdb;ipdb.set_trace()
    # acc = ((pred >= 0.5) == gt_label).mean()
    # fpr, tpr, _ = roc_curve(gt_label, pred)
    # roc_auc = auc(fpr, tpr)
    # return ap, acc, roc_auc

    return result



