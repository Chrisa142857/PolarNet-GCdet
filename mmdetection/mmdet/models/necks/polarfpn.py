from ..builder import NECKS
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16


import torch
import torch.nn as nn

import torch.nn.functional as F
import random

class PolarNet(torch.nn.Module):
    def __init__(self, hidden_num=256, polar_repeatnum=1):
        super().__init__()
        self.hidden_num = hidden_num
        self.mask_id_cache = {}
        self.polar_repeatnum = polar_repeatnum
        self.polar_conv1 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        self.polar_conv2 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        self.polar_conv3 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        # self.polar_conv1 = nn.ModuleList()
        # self.polar_conv2 = nn.ModuleList()
        # self.polar_conv3 = nn.ModuleList()
        # # self.polar_conv4 = nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1)
        # for i in range(self.polar_repeatnum):
        #     self.polar_conv1.append(nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1))
        #     self.polar_conv2.append(nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1))
        #     self.polar_conv3.append(nn.Conv2d(in_channels=hidden_num, out_channels=hidden_num, kernel_size=1, stride=1))
        # self.polar_conv3 = None
        self.norm_layer = True

    def generate_8neib_id(self, H, W, device):
        if '%d-%d' % (H, W) not in self.mask_id_cache:
            ids = torch.LongTensor([i for i in range(H*W)]).to(device).reshape(H, W)
            x_id = torch.LongTensor([
                [i-1 for _ in range(3)] + 
                [i for _ in range(3)] + 
                [i+1 for _ in range(3)]
                for i in range(H)
            ]).clamp(min=0, max=H-1).to(device)
            y_id = torch.LongTensor([
                [i-1, i, i+1] * 3
                for i in range(W)
            ]).clamp(min=0, max=W-1).to(device)
            mask_id = []
            # nei_keys = []
            # nei_vectors = []
            for i in range(H):
                for j in range(W):
                    mask_id.append(ids[x_id[i], y_id[j]])
                    # nei_keys.append(torch.index_select(k, 2, ids[x_id[i], y_id[j]]))
                    # if self.polar_conv3 != None:
                    #     nei_vectors.append(torch.index_select(v, 2, ids[x_id[i], y_id[j]]))
            mask_id = torch.cat(mask_id)
            self.mask_id_cache['%d-%d' % (H, W)] = mask_id
        else:
            mask_id = self.mask_id_cache['%d-%d' % (H, W)]
        return mask_id

    def get_polarity(self, x, i=0):
        B, C, H, W = x.shape
        if i != None:
            polar_conv1 = self.polar_conv1[i]
            polar_conv2 = self.polar_conv2[i]
            polar_conv3 = self.polar_conv3[i]
        else:
            polar_conv1 = self.polar_conv1
            polar_conv2 = self.polar_conv2
            polar_conv3 = self.polar_conv3
        q = polar_conv1(x).reshape(B, C, H*W).permute(0, 2, 1)
        q = q.unsqueeze(2) # B x N x 1 x C, N=H*W
        k = polar_conv2(x).reshape(B, C, H*W)
        if self.polar_conv3 != None:
            v = polar_conv3(x).reshape(B, C, H*W)
        mask_id = self.generate_8neib_id(H, W, x.device)
        nei_keys = torch.index_select(k, 2, mask_id).reshape(B, C, H*W, 9)
        nei_keys = nei_keys.permute(0, 2, 1, 3) # B x N x C x 9, N=H*W
        if self.polar_conv3 != None:
            nei_vectors = torch.index_select(v, 2, mask_id).reshape(B, C, H*W, 9)
        nei_vectors = nei_vectors.permute(0, 2, 3, 1) # B x N x 9 x C, N=H*W

        out = torch.einsum('bnpc,bncq->bnpq', q, nei_keys) # B x N x 1 x 9, N=H*W, P=1, Q=9 
        if self.polar_conv3 != None:
            new_v = torch.einsum('bnpq,bnqc->bnpc', out, nei_vectors) # B x N x 1 x C, N=H*W, P=1, Q=9 
            new_v = new_v.squeeze(dim=2).permute(0, 2, 1) # B, C, H*W
            new_v = new_v.reshape(B, C, H, W)
        out = out.squeeze(dim=2) # B, H*W, 9
        out = out.reshape(B, H, W, 9)  # B, H, W, 9
        scores = F.softmax(out, dim=-1) # B, H, W, 9
        polarity = scores.argmax(dim=-1) # B, H, W
        output = torch.stack([
            out[..., 4], 
            torch.cat([
                out[..., :4].sum(dim=-1, keepdim=True), 
                out[..., 5:].sum(dim=-1, keepdim=True)
                ], dim=-1).sum(-1) / 8
                ], dim=-1) # B, H, W, 2

        if self.polar_conv3 != None:
            x = x + new_v
            if self.norm_layer:
                x = F.layer_norm(x, [H, W])
        return output, x
    
    def forward(self, x):
        # for i in range(self.polar_repeatnum):
        #     polar, x = self.get_polarity(x, i)
        polar, x = self.get_polarity(x, None)
        return polar, x
  

@NECKS.register_module()
class EmptyNeck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),     
                polar_in_index=-1,
                polar_repeatnum=1):
        super(EmptyNeck, self).__init__(init_cfg)
        
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        outs = inputs[-1]     
        return tuple([outs])

@NECKS.register_module()
class PolarNeck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),     
                polar_in_index=-1,
                polar_repeatnum=1):
        super(PolarNeck, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.out_channels = out_channels
        self.polar_net = PolarNet(out_channels, polar_repeatnum)
        self.polar_in_index = polar_in_index 
        
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        outs = [inputs[i] for i in range(len(inputs))]
        polar_scores, polar_fm = self.polar_net(outs[self.polar_in_index])
        outs[self.polar_in_index] = polar_fm
        return tuple([outs[self.polar_in_index]]), polar_scores


@NECKS.register_module()
class PolarFPN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),     
                polar_in_index=-1,
                polar_repeatnum=1):
        super(PolarFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.polar_net = PolarNet(out_channels, polar_repeatnum)
        self.polar_in_index = polar_in_index

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        polar_scores, polar_fm = self.polar_net(outs[self.polar_in_index])
        outs[self.polar_in_index] = polar_fm
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        return tuple(outs), polar_scores
