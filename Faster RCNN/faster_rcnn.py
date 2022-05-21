#!/usr/bin/env python
# coding=utf-8
"""
Faster R-CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Dict, List, Tuple, Optional
from torch import Tensor

from utils.pooler import NewMultiScaleRoIAlign
from utils.generalized_rcnn import GeneralizedRCNN
from utils.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from utils.roi_head import RoIHeads
from utils.transform import GeneralizedRCNNTransform


from datasets import transforms as T
from torchvision import transforms

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
  


class FasterRCNN(GeneralizedRCNN):

    def __init__(self, backbone, num_classes=None,
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000,
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000,
                 rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_img=256,
                 rpn_positive_fraction=0.5,
                 box_roi_pool=None,
                 box_head=None,
                 box_predictor=None,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_img=512,
                 box_positive_fraction=0.25,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 # box_detections_per_img=20,
                 bbox_reg_weights=None):
        
        if not hasattr(backbone, "out_channels"):
            raise ValueError("backbone should have the out_channels attr")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        # assert isinstance(box_roi_pool, (NewMultiScaleRoIAlign, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,),(64,),(128,),(256,),(512,))
            # anchor_sizes = ((32,64,128),(32,64,128),(64,128,256),(64,128,256),(128,256,512))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            # anchor_sizes = ((64,),(128,),(192,),(256,),(320,))
            # aspect_ratios = ((0.5, 1.0, 4.0),(0.5, 1.0, 3.5),(0.5, 1.0, 5.0),(0.5, 1.0, 3.0),(0.5, 1.0, 4.0))
            # anchor_sizes = ((128, 256, 512),)
            # aspect_ratios = ((0.5, 1.0, 2.0),)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes,
                                                   aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels,
                               rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train,
            testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train,
            testing=rpn_post_nms_top_n_test
        )
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n, rpn_post_nms_top_n,
            rpn_nms_thresh, rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_img, rpn_positive_fraction
        )
        
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=7,
                sampling_ratio=2
            )
        # if box_roi_pool is None:
        #     box_roi_pool = MultiScaleRoIAlign(
        #         featmap_names=["3"],
        #         output_size=7,
        #         sampling_ratio=2
        #     )
        # if box_roi_pool is None:
        #     box_roi_pool = MultiScaleRoIAlign(
        #         featmap_names=["3"],
        #         output_size=7,
        #         sampling_ratio=2
        #     )
            # box_roi_pool = NewMultiScaleRoIAlign(
            #     featmap_names=["0", "1", "2", "3"],
            #     output_size=7,
            #     sampling_ratio=2
            # )
        if box_head is None:
            # roi特征的尺寸
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLHead(
                out_channels * resolution ** 2,
                representation_size
            )
            # box_head = NewTwoMLHead(
            #     out_channels,
            #     resolution,
            #     representation_size
            # )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes=num_classes
            )

        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_img, box_positive_fraction,
            box_score_thresh, box_nms_thresh,
            box_detections_per_img,
            bbox_reg_weights
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size,
                                             image_mean, image_std)
        super(FasterRCNN, self).__init__(transform, backbone, rpn, roi_heads)


class TwoMLHead(nn.Module):
    
    def __init__(self, in_channels, representation_size):
        super(TwoMLHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class NewTwoMLHead(nn.Module):

    def __init__(self, in_channels, roi_size, representation_size):
        super(NewTwoMLHead, self).__init__()
        self.fc6 = nn.ModuleList()
        num_levels = 4
        for i in range(num_levels):
            self.fc6.append(
                nn.Sequential(
                    nn.Linear(in_channels*roi_size**2, representation_size),
                    nn.GroupNorm(32, representation_size, 1e-5),
                    nn.ReLU(inplace=True)
                )
            )

        self.fc7 = nn.Sequential(
            nn.Linear(representation_size, representation_size),
            nn.GroupNorm(32, representation_size, 1e-5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x[0].shape[0]
        for i in range(len(x)):
            x[i] = self.fc6[i](x[i].view(batch_size, -1))
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.fc7(x)
        return x


class FastRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes*4)
        # nn.init.normal_(self.cls_score.weight, 0., 0.01)
        # nn.init.constant_(self.cls_score.bias, -2.0)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class TraceFasterRCNN(FasterRCNN):

    def forward(self, images, targets=None):

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            img_size = img.shape[-2:]
            assert len(img_size) == 2
            original_image_sizes.append((img_size[0], img_size[1]))
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes,
                                                     targets)
        detections = self.transform.postprocess(detections,
                                                images.image_sizes,
                                                original_image_sizes)

        # print(detections[0]['labels'].unsqueeze(1).shape, detections[0]['boxes'].shape)
        
        out = []
        for det in detections:
            tail = torch.stack([torch.ones_like(det['scores']), torch.zeros_like(det['scores']), torch.zeros_like(det['scores'])], dim=-1)
            # print("det['scores'].shape", det['scores'].shape)
            # print('tail.shape',tail.shape)
            bw = det['boxes'][:, 2] - det['boxes'][:, 0]
            bh = det['boxes'][:, 3] - det['boxes'][:, 1]
            boxes = torch.stack([
                det['boxes'][:, 0] + bw/2,
                det['boxes'][:, 1] + bh/2,
                bw,
                bh
                ], dim=1)
            for i in range(det['boxes'].shape[0]):
                tail[i, det['labels'][i].long()] = det['scores'][i]
            out.append(torch.cat([boxes, tail], dim=1))
        
        return [torch.stack(out, dim=0), torch.zeros_like(out[0])]

if __name__ == "__main__":
    from utils.backbone_utils import resnet_fpn_backbone, densenet_fpn_backbone
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    pth_name = r'D:\WSI_analysis\det_train\AttFaster-glandular\results\densenet161\latest\densenet169.pth'
    # save_pt_name = pth_name.replace('.pth','.pt')
    bsize = 16
    isize = 1024
    save_pt_name = 'detector_densenet161_input%dx3x%dx%d.pt' % (bsize, isize, isize)
    backbone = densenet_fpn_backbone("densenet161", True)
    model = TraceFasterRCNN(backbone, num_classes=3)
    model.trace_module = True
    model.load_state_dict(torch.load(pth_name, map_location='cpu'))
    print(model)
    T = get_transform(False)
    from PIL import Image
    x = T(Image.open(r'E:\wei\glandular1536_20211214\images\1M09.mrxs_x28635_y124183_side1536.jpg'))
    x= torch.stack([x for _ in range(bsize)], dim=0).cuda()
    print('x.shape',x.shape)
    model = model.eval().cuda()
    with torch.no_grad():
        # y=model(x)
        # print(type(y), y[0].shape)
        # exit()
        traced_script_module = torch.jit.trace(model, x, check_trace=False)
        traced_script_module.save(save_pt_name)
        y=traced_script_module(x)
        print(y[0])
        print(type(y), y[0].shape)
        # dis = y[0]-y1[0]
        # print(dis.mean(), dis.max(), dis.min())
    # import ipdb;ipdb.set_trace()

