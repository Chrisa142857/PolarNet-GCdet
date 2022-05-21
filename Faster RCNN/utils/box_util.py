#!/usr/bin/env python
# coding=utf-8
import torch
from torch.jit.annotations import List, Tuple

import math


def zeros_like(tensor, dtype):
    # type: (Tensor, int) -> Tensor
    return torch.zeros_like(tensor, dtype=dtype, layout=tensor.layout,
                            device=tensor.device,
                            pin_memory=tensor.is_pinned())


@torch.jit.script
class BalancedPositiveNegativeSampler(object):

    def __init__(self, batch_size_per_img, positive_fraction):
        # type: (int, float)
        self.batch_size_per_img = batch_size_per_img
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor])
        pos_idx = []
        neg_idx = []

        for matched_idxs_per_img in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_img >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_img == 0).squeeze(1)

            num_pos = int(self.batch_size_per_img * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_img - num_pos
            num_neg = min(negative.numel(), num_neg)
            
            perm1 = torch.randperm(
                positive.numel(), device=positive.device
            )[:num_pos]
            perm2 = torch.randperm(
                negative.numel(), device=negative.device
            )[:num_neg]

            pos_idx_per_img = positive[perm1]
            neg_idx_per_img = negative[perm2]

            pos_idx_per_img_mask = zeros_like(
                matched_idxs_per_img, dtype=torch.uint8
            )
            neg_idx_per_img_mask = zeros_like(
                matched_idxs_per_img, dtype=torch.uint8
            )

            pos_idx_per_img_mask[pos_idx_per_img] = torch.tensor(
                1, dtype=torch.uint8, device=matched_idxs_per_img.device
            )
            neg_idx_per_img_mask[neg_idx_per_img] = torch.tensor(
                1, dtype=torch.uint8, device=matched_idxs_per_img.device
            )

            pos_idx.append(pos_idx_per_img_mask)
            neg_idx.append(neg_idx_per_img_mask)

        return pos_idx, neg_idx


@torch.jit.script
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    anchor_widths = proposals_x2 - proposals_x1
    anchor_heights = proposals_y2 - proposals_y1
    anchor_ctr_x = proposals_x1 + 0.5 * anchor_widths
    anchor_ctr_y = proposals_y1 + 0.5 * anchor_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    dx = wx * (gt_ctr_x - anchor_ctr_x) / anchor_widths
    dy = wy * (gt_ctr_y - anchor_ctr_y) / anchor_heights
    dw = ww * torch.log(gt_widths / anchor_widths)
    dh = wh * torch.log(gt_heights / anchor_heights)

    targets = torch.cat((dx, dy, dw, dh), dim=1)
    return targets


@torch.jit.script
class BoxCoder(object):

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float)
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor])
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, dim=0)

    def encode_single(self, reference_boxes, proposals):
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes.cpu(), proposals.cpu(), weights.cpu())
        return targets.to(device)

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor])
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)

        boxes_per_image = [b.shape[0] for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        pred_boxes = self.decode_single(
            rel_codes.reshape(box_sum, -1),
            concat_boxes
        )
        return pred_boxes.reshape(box_sum, -1, 4)

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)
        
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_x1 = pred_ctr_x - torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device
        ) * pred_w
        pred_y1 = pred_ctr_y - torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device
        ) * pred_h
        pred_x2 = pred_ctr_x + torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device
        ) * pred_w
        pred_y2 = pred_ctr_y + torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device
        ) * pred_h

        pred_boxes = torch.stack(
            (pred_x1, pred_y1, pred_x2, pred_y2), dim=2
        ).flatten(1)
        return pred_boxes


@torch.jit.script
class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    
    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold, low_threshold,
                 allow_low_quality_matches=False):
        # type: (float, float, bool)
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2

    def __call__(self, match_quality_matrix):
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        device = match_quality_matrix.device
        matches[below_low_threshold] = torch.tensor(self.BELOW_LOW_THRESHOLD).to(device)
        matches[between_thresholds] = torch.tensor(self.BETWEEN_THRESHOLDS).to(device)

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(
                matches, all_matches, match_quality_matrix
            )

        return matches


    def set_low_quality_matches_(self, matches, all_matches,
                                 match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


if __name__ == "__main__":
    import torch


    box_coder = BoxCoder(weights=[1.,1.,1.,1.])
    rel_boxes = torch.arange(0, 16, dtype=torch.float32).reshape(2, -1)
    boxes = [torch.arange(1, 5, dtype=torch.float32).reshape(1, -1),
             torch.arange(6, 10, dtype=torch.float32).reshape(1, -1)]

    pred_boxes = box_coder.decode(rel_boxes, boxes)
    import ipdb;ipdb.set_trace()
