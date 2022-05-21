#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from polar_net import polarity_loss, get_mask
from torchvision.ops import roi_align
import torch.nn.functional as F

from yolox.utils.boxes import postprocess

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

celoss = nn.CrossEntropyLoss()

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, postprocess=False, fg_net=None, polar_net=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        
        self.fg_net = fg_net
        self.polar_net = polar_net
        if polar_net != None:
            if 'a' in polar_net.mtype:
                self.feat_k = 1
            elif 'b' in polar_net.mtype:
                self.feat_k = 0
            else:
                self.feat_k = 2
        self.backbone = backbone
        self.head = head
        self.postprocess = postprocess
        if postprocess:
            self.num_classes=2
            self.confthre=0.1
            self.nmsthre=0.5

    def forward(self, x, targets=None, return_pscore=False):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        if self.polar_net != None:
            # v4 #############
            polarity, fpn_outs[self.feat_k] = self.polar_net(fpn_outs[self.feat_k])
            polarity, p_score = polarity[:2]
            ##################
            # p_loss = (polarity!=4).float().mean()
            # p_num = 1
            # for box_p, box_l in zip(box_polarities, labels):
            #     assert box_p.shape[0] == box_l.shape[0]
            #     p_loss += polarity_loss(box_p, box_l)
            #     p_num += 1
            # p_loss /= p_num
            ################
        if self.fg_net != None:
            fg_net_outputs, fg_scores = self.fg_net(x, self.head.strides, fpn_outs)
        else:
            fg_scores = None
        p_scores = None
        if self.training:
            assert targets is not None

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            
            if fg_scores != None:
                fg_loss = self.fg_net.get_loss(fg_net_outputs, targets)
                loss += fg_loss
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                    "fg_loss": fg_loss,
                }
            elif self.polar_net != None:
                fg_masks, bg_masks = get_mask(polarity, self.head.strides, targets, self.feat_k)
                p_loss = celoss(p_score.view(-1, 2), fg_masks.long().view(-1))
                loss += p_loss
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                    "polar_loss": p_loss,
                }
            else:
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
        else:
            self.head.decode_in_inference = True
            outputs = self.head(fpn_outs, fg_scores=fg_scores)
            B = outputs.shape[0]
            # new_outputs = []
            # for bi in range(B):
            #     conf, _ = torch.max(outputs[bi, :, 5:7], dim=1)
            #     # scores = outputs[bi, :, 4] * conf
            #     new_id = torch.argsort(conf, descending=True)
            #     # outputs = torch.stack([outputs[bi, new_id[bi, :2000], :] ])
            #     new_outputs.append(outputs[bi, new_id[:2000], :])
            # outputs = torch.stack(new_outputs)
            if self.fg_net != None:
                outputs = [outputs, fg_scores]
            elif self.polar_net != None:
                # post_outputs = postprocess(
                #     outputs, self.num_classes, self.confthre, self.nmsthre
                # )
                # boxes = [out[:, :4] if out != None else torch.zeros(0, 4).dtype(post_outputs.type()) for out in post_outputs]
                # box_x1 = outputs[..., 0] - outputs[..., 2]/2
                # box_y1 = outputs[..., 1] - outputs[..., 3]/2
                # box_x2 = outputs[..., 0] + outputs[..., 2]/2
                # box_y2 = outputs[..., 1] + outputs[..., 3]/2
                # # boxes = torch.stack([box_x1, box_y1, box_x2, box_y2], dim=-1)
                # boxes = [torch.stack([x1, y1, x2, y2], -1) for x1, y1, x2, y2 in zip(box_x1, box_y1, box_x2, box_y2)]

                # ##################
                # box_polarities = roi_align(
                #     p_score, 
                #     boxes, 
                #     output_size=7, 
                #     spatial_scale=1/self.head.strides[self.feat_k]
                #     )

                # p_scores = F.max_pool2d(box_polarities, 7)[:, :, 0, 0]
                # # p_scores = F.avg_pool2d(box_polarities, 7)[:, :, 0, 0]

                # p_score_o = F.softmax(p_scores, dim=1)[:, 1]

                p_scores = torch.zeros_like(outputs)[:, :, 0]
                # N = outputs.shape[1]
                # # outputs[:, :, 4] = (outputs[:, :, 4] + p_score_o[:B].unsqueeze(1).repeat(1, N)) / 2
                # for bi in range(B):
                #     # outputs[bi, :, 4] = outputs[bi, :, 4] * p_score_o[bi*N:(bi+1)*N]
                #     outputs[bi, :, 4] = (outputs[bi, :, 4] + p_score_o[bi*N:(bi*N+1)]) / 2
                #     p_scores[bi, :] = p_score_o[bi:(bi+1)]
                #     # p_scores[bi, :] = p_score_o[bi*N:(bi+1)*N]
                ##################
                for bi in range(B):
                    box_x1 = outputs[bi, :, 0] - outputs[bi, :, 2]/2
                    box_y1 = outputs[bi, :, 1] - outputs[bi, :, 3]/2
                    box_x2 = outputs[bi, :, 0] + outputs[bi, :, 2]/2
                    box_y2 = outputs[bi, :, 1] + outputs[bi, :, 3]/2
                    box_polarities = roi_align(
                        p_score[bi:bi+1], 
                        [torch.stack([box_x1, box_y1, box_x2, box_y2], -1)], 
                        output_size=7, 
                        spatial_scale=1/self.head.strides[self.feat_k]
                        )
                    box_polarities = F.max_pool2d(box_polarities, 7)[:, :, 0, 0]
                    p_score_o = F.softmax(box_polarities, dim=1)[:, 1]
                    outputs[bi, :, 4] = (outputs[bi, :, 4] + p_score_o[0:1]) / 2
                    p_scores[bi, :] = p_score_o[0:1]
                ##################
            #     post_outputs = postprocess(
            #         outputs, self.num_classes, self.confthre, self.nmsthre
            #     )
            #     if post_outputs[0] != None:
            #         outputs = torch.stack(post_outputs, dim=0)
            #         outputs = torch.cat([outputs[:, :, 0:4], outputs[:, :, 6:7], outputs[:, :, 4:5] * outputs[:, :, 5:6]], dim=2)
            #         outputs = [outputs[:, :100, :], torch.empty(0)]
        if return_pscore and p_scores != None:
            return outputs, p_scores
        
        if self.postprocess:
            outputs = [outputs, torch.empty_like(outputs)]
            
        return outputs
