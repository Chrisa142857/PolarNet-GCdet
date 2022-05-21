#!/usr/bin/env python
# coding=utf-8
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import List, Optional, Dict
from . import box_ops
from . import box_util
from .focalloss import BCEFocalLoss


bcefocalloss = BCEFocalLoss()


class AnchorGenerator(nn.Module):

    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }
    
    def __init__(self,
                 sizes=(128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios,
                         dtype=torch.float32, device="cpu"):
        # type: (List[int], List[float], int, Device) # noqa: F821
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype,
                                        device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        # type: (int, Device) -> None    # noqa: F821
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                dtype,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]])
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        for (size, stride, base_anchors) in zip(grid_sizes,
                                                strides,
                                                cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            shift_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shift_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y),
                                 dim=1)
            anchors.append(
                (shifts.view(-1, 1, 4)+base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]])
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor])
        grid_sizes = list([feature_map.shape[-2:]
                          for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # strides = [[torch.tensor(image_size[0]/g[0],
        #                          dtype=dtype, device=device),
        #            torch.tensor(image_size[1]/g[1],
        #                         dtype=dtype, device=device)]
        #           for g in grid_sizes]
        strides = [[image_size[0]/g[0], image_size[1]/g[1]] for g in grid_sizes]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(
            grid_sizes, strides
        )
        anchors = torch.jit.annotate(List[List[Tensor]], [])
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        anchors = [torch.cat(anchors_per_image)
                  for anchors_per_image in anchors]
        self._cache.clear()
        return anchors


class RPNHead(nn.Module):

    def __init__(self, in_channels, num_anchors_per_loc):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors_per_loc,
                                    kernel_size=1, stride=1)
        nn.init.normal_(self.cls_logits.weight, mean=0., std=0.01)
        # nn.init.constant_(self.cls_logits.bias, -2.0)
        nn.init.constant_(self.cls_logits.bias, 0.)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors_per_loc*4,
                                   kernel_size=1, stride=1)
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0.)

        # for l in self.children():
        #     nn.init.normal_(l.weight, std=0.01)
        #     nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # type: (List[Tensor])
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int)
    layer = layer.view(N, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor])
    box_cls_flatten = []
    box_regression_flatten = []
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AXC, H, W = box_cls_per_level.shape
        AX4 = box_regression_per_level.shape[1]
        A = AX4 // 4
        C = AXC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flatten.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flatten.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flatten, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flatten, dim=1).reshape(-1,4)
    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):

    __annotations__ = {
        "box_coder": box_util.BoxCoder,
        "proposal_matcher": box_util.Matcher,
        "fg_bg_sampler": box_util.BalancedPositiveNegativeSampler,
        "pre_nms_top_n": Dict[str, int],
        "post_nms_top_n": Dict[str, int],
    }

    def __init__(self,
                 anchor_generator,
                 head,
                 pre_nms_top_n,
                 post_nms_top_n,
                 nms_thresh,
                 fg_iou_thresh,
                 bg_iou_thresh,
                 batch_size_per_img,
                 positive_fraction):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = box_util.BoxCoder(weights=(1., 1., 1., 1.))

        self.proposal_matcher = box_util.Matcher(
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = box_util.BalancedPositiveNegativeSampler(
            batch_size_per_img, positive_fraction
        )

        # 预测框筛选时要保留的框数量的设置
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]])
        labels = []
        matched_gt_boxes = []
        for anchors_per_img, targets_per_img in zip(anchors, targets):
            gt_boxes = targets_per_img["boxes"]

            device = anchors_per_img.device
            if gt_boxes.numel() == 0:
                matched_gt_boxes_per_img = torch.zeros(
                    anchors_per_img.shape, dtype=torch.float32,
                    device=device
                )
                labels_per_img = torch.zeros(
                    (anchors_per_img.shape[0],), dtype=torch.float32,
                    device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes, anchors_per_img
                )
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_img = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_img = matched_idxs >= 0
                labels_per_img = labels_per_img.to(dtype=torch.float32)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_img[bg_indices] = torch.tensor(0.0).to(device)
                indices_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_img[indices_to_discard] = torch.tensor(-1.0).to(device)

            labels.append(labels_per_img)
            matched_gt_boxes.append(matched_gt_boxes_per_img)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int])
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, dim=1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness,
                         image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int])
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                 for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        num_topn_images = proposals.shape[0]

        final_boxes = []
        final_scores = []
        for i in range(num_topn_images):
            boxes, scores, lvl, image_shape = proposals[i], objectness[i], levels[i], image_shapes[i]
        # for boxes, scores, lvl, image_shape in zip(proposals, objectness,
        #                                            levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas,
                     labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor])
        sampled_pos_idx_mask, sampled_neg_idx_mask = self.fg_bg_sampler(
            labels
        )
        sampled_pos_idxs = torch.nonzero(
            torch.cat(sampled_pos_idx_mask, dim=0)
        ).squeeze(1)
        sampled_neg_idxs = torch.nonzero(
            torch.cat(sampled_neg_idx_mask, dim=0)
        ).squeeze(1)
        sampled_idxs = torch.cat([sampled_pos_idxs, sampled_neg_idxs],
                                 dim=0)

        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)

        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.l1_loss(
            pred_bbox_deltas[sampled_pos_idxs],
            regression_targets[sampled_pos_idxs],
            reduction="sum"
        ) / (sampled_idxs.numel())
        # 分类损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_idxs], labels[sampled_idxs]
        )
        # objectness_loss = bcefocalloss(
        #     objectness[sampled_idxs], labels[sampled_idxs]
        # )

        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        # type: (ImageList, Dict[str, Tensor], Optional[List[Dict[str, Tensor]]])
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape
                                              for o in objectness]
        num_anchors_per_level = [s[0]*s[1]*s[2]
                                for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(),
                                          anchors)
        proposals = proposals.reshape(num_images, -1, 4)
        boxes, scores = self.filter_proposals(
            proposals, objectness,
            images.image_sizes, num_anchors_per_level
        )

        losses = {}

        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(
                anchors, targets
            )
            regression_targets = self.box_coder.encode(
                matched_gt_boxes, anchors
            )
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses


if __name__ == "__main__":
    import torch
    from backbone_utils import resnet_fpn_backbone

    image = torch.randn(1, 3, 224, 224)

    net = resnet_fpn_backbone("resnet50", True)
    features = net(image)
    features = list(features.values())

    rpn = RPNHead(256, 3)
    logits, bbox_reg = rpn(features)
    import ipdb;ipdb.set_trace()

