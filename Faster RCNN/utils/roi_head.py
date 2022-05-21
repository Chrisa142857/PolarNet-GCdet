#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.jit.annotations import List, Dict, Optional, Tuple

from . import box_ops
from . import box_util
from .focalloss import CEFocalLoss

from polar_net import polarity_loss, get_mask
from torchvision.ops import roi_align

cefocalloss = CEFocalLoss(class_nums=2)
celoss = nn.CrossEntropyLoss()

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    # classification_loss = cefocalloss(class_logits, labels)
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum"
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class RoIHeads(nn.Module):

    __annotations__ = {
        "proposal_matcher": box_util.Matcher,
        "fg_bg_sampler": box_util.BalancedPositiveNegativeSampler,
        "box_coder": box_util.BoxCoder,
    }
    def __init__(self,
                 # RoIAlign
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_img,
                 positive_fraction,
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 bbox_reg_weights=None):
        super(RoIHeads, self).__init__()

        self.proposal_matcher = box_util.Matcher(
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=False
        )
        self.fg_bg_sampler = box_util.BalancedPositiveNegativeSampler(
            batch_size_per_img, positive_fraction
        )
        
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = box_util.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def subsample(self, labels):
        # type: (List[Tensor])
        sampled_pos_masks, sampled_neg_masks = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_mask, neg_mask) in enumerate(
            zip(sampled_pos_masks, sampled_neg_masks)
        ):
            img_sampled_inds = torch.nonzero(pos_mask | neg_mask).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor])
        matched_idxs = []
        labels = [] 
        for proposals_in_img, gt_boxes_in_img, gt_labels_in_img in zip(
            proposals, gt_boxes, gt_labels
        ):
            device = proposals_in_img.device
            if gt_boxes_in_img.numel() == 0:
                clamped_matched_idxs_in_img = torch.zeros(
                    (proposals_in_img.shape[0],), dtype=torch.int64,
                    device=device
                )
                labels_in_img = torch.zeros(
                    (proposals_in_img.shape[0],), dtype=torch.int64,
                    device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_img, proposals_in_img
                )
                matched_idxs_in_img = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_img = matched_idxs_in_img.clamp(
                    min=0
                )
                labels_in_img = gt_labels_in_img[clamped_matched_idxs_in_img]
                labels_in_img = labels_in_img.to(dtype=torch.int64)
                bg_indices = matched_idxs_in_img == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_img[bg_indices] = torch.tensor(0).to(device)
                ignore_indices = matched_idxs_in_img == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_img[ignore_indices] = torch.tensor(-1).to(device)

            matched_idxs.append(clamped_matched_idxs_in_img)
            labels.append(labels_in_img)
        return matched_idxs, labels

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor])
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            gt_boxes_in_img = gt_boxes[img_id]
            if gt_boxes_in_img.numel() == 0:
                gt_boxes_in_img = torch.zeros((1, 4), dtype=dtype,
                                              device=device)
            matched_gt_boxes.append(gt_boxes_in_img[matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression,
                               proposals, image_shapes, polarity=None, gt_labels=None):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_img.shape[0]
                           for boxes_in_img in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        if polarity != None:
            polarity_list = polarity.split(boxes_per_image, 0)
        if gt_labels != None:
            gt_list = gt_labels.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_polarity = []
        all_gt = []
        for i, (boxes, scores, image_shape) in enumerate(zip(
            pred_boxes_list, pred_scores_list, image_shapes
        )):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            boxes = boxes[:, 1:]
            labels = labels[:, 1:]
            scores = scores[:, 1:]

            clsnum = boxes.shape[1]
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            if polarity != None:
                p = polarity_list[i]
                p = torch.stack([p for _ in range(clsnum)], dim=1)
                p = p.reshape(-1, 2, p.shape[-2], p.shape[-1])
            if gt_labels != None:
                l = gt_list[i]
                l = torch.stack([l for _ in range(clsnum)], dim=1)
                l = l.reshape(-1)

            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            if polarity != None:
                p = p[inds]
            if gt_labels != None:
                l = l[inds]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if polarity != None:
                p = p[keep]
            if gt_labels != None:
                l = l[keep]

            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if polarity != None:
                p = p[keep]
            if gt_labels != None:
                l = l[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            if polarity != None:
                all_polarity.append(p)
            if gt_labels != None:
                all_gt.append(l)

        if gt_labels != None:
            return all_boxes, all_scores, all_labels, all_polarity, all_gt
        if polarity != None:
            return all_boxes, all_scores, all_labels, all_polarity
        return all_boxes, all_scores, all_labels
            
    def forward(self, features, proposals, image_shapes, polarity=None, polar_net=None, topk=None, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        
        if 'a' in polar_net.mtype: 
            roi_align_scale = 16
        elif 'b' in polar_net.mtype: 
            roi_align_scale = 8
        elif 'c' in polar_net.mtype: 
            roi_align_scale = 4
        else: 
            roi_align_scale = 32

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        # v4 #############
        if polarity != None:
            polarity, p_score = polarity[:2]
            # box_polarities = roi_align(
            #     p_score, 
            #     proposals, 
            #     output_size=self.box_roi_pool.output_size, 
            #     spatial_scale=self.box_roi_pool.scales[-1]
            #     )
        ################
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {} 
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )

            if polarity != None:
                ###############################
                fg_masks, bg_masks = get_mask(polarity, image_shapes=image_shapes, targets=targets)
                # fg_masks = torch.stack((fg_masks, fg_masks), -1)
                B = polarity.shape[0]
                # print(p_score.view(B, -1, 2).shape, fg_masks.long().shape)
                p_loss = celoss(p_score.view(-1, 2), fg_masks.long().view(-1))
                ###############################
                # pred_boxes, scores, pred_labels, box_polarities, labels = self.postprocess_detections(
                #     class_logits, box_regression, proposals, image_shapes, polarity=box_polarities, gt_labels=torch.cat(labels, dim=0)
                # )
                
                # p_loss = (polarity!=4).float().mean()
                # p_num = 1
                # for box_p, box_l in zip(box_polarities, labels):
                #     assert box_p.shape[0] == box_l.shape[0]
                #     p_loss += polarity_loss(box_p, box_l)
                #     p_num += 1
                # p_loss /= p_num
                ###############################
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_polarity": p_loss
            }
            # if torch.isnan(loss_classifier) or torch.isnan(loss_box_reg) or loss_classifier > 100 or loss_box_reg > 10:
            #     import ipdb;ipdb.set_trace()

        else:

            if polarity != None:
                box_polarities = roi_align(
                    p_score, 
                    proposals, 
                    output_size=7, 
                    spatial_scale=1/roi_align_scale
                    )
                pred_boxes, scores, pred_labels, box_polarities = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes, polarity=box_polarities
                )
            else:
                pred_boxes, scores, pred_labels = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes
                )
            num_images = len(pred_boxes)
            for i in range(num_images):
                if polarity != None and 'v4' in polar_net.mtype:
                    box_polarity = box_polarities[i]
                    if len(box_polarity) > 0:
                        p_scores = F.max_pool2d(box_polarity, box_polarity.shape[-1])[:, :, 0, 0]
                        p_score_o = F.softmax(p_scores)[:, 1]
                        # p_score_o = p_scores[:, 1]
                    else:
                        p_score_o = torch.empty_like(scores[i])

                result.append(
                    {
                        "boxes": pred_boxes[i],
                        "labels": pred_labels[i],
                        "scores": (scores[i] + p_score_o) / 2,
                        # "scores": scores[i],
                        "p_score": p_score_o
                    }
                )
                    
        return result, losses

####
def vis(polarity):
    import matplotlib.pyplot as plt
    scl = 3
    direct = [0,1,2,3,-1,3,2,1,0]
    dx = [[-1, 1], [0, 0], [1, -1], [-1, 1]]
    dy = [[-1, 1], [-1, 1], [-1, 1], [0, 0]]
    plt.figure(figsize=(polarity.shape[-2]/3,polarity.shape[-1]/3))
    for i in range(polarity.shape[-2]):
        for j in range(polarity.shape[-1]):
            d = direct[polarity[0, i, j]]
            x1 = i*scl + dx[d][0]*scl-1
            y1 = j*scl + dy[d][0]*scl-1
            x2 = i*scl + dx[d][1]*scl-1
            y2 = j*scl + dy[d][1]*scl-1
            plt.plot([x1,x2], [y1,y2], '-', color='black')
    plt.savefig('temp.jpg')
