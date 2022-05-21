#!/usr/bin/env python
# coding=utf-8
"""
Channel attention and Spatial attention
"""
import torch
from torch import nn
import torch.nn.functional as F


class Flatten_layer(nn.Module):
    """
    used in Channel attention module to flatten the Cx1x1 tensor
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    """
    perform the channel attention. we take the high-level feature
    as input to produce the channel attention map to guide the low-level
    feature to highlight important channels and supress unimportant ones 
    """

    def __init__(self, in_channels, reduction_ratios=16,
                 pool_types=["avg", "max"]):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        # the shared multi-layer perceptron
        self.mlp = nn.Sequential(
            Flatten_layer(),
            nn.Linear(in_channels, in_channels // reduction_ratios),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratios, in_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_attention_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                raw_channel_attention = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)),
                                        stride=(x.size(2), x.size(3)))
                raw_channel_attention = self.mlp(max_pool)

            if channel_attention_sum is None:
                channel_attention_sum = raw_channel_attention
            else:
                channel_attention_sum = channel_attention_sum + raw_channel_attention

        scale = F.sigmoid(channel_attention_sum).unsqueeze(2).unsqueeze(3)
        return scale


class ChannelPool(nn.Module):
    """
    do pooling op along the channel axis
    """
    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, avg_pool), dim=1)


class ConvModule(nn.Module):
    """
    do conv op after the channelpool
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=3, relu=True, bias=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    """
    spatial attention module. take channel attention results as input
    """

    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.channelpool = ChannelPool()
        self.convlayer = ConvModule(2, 1, 7, relu=False)

    def forward(self, x):
        x_channelpool = self.channelpool(x)
        x_out = self.convlayer(x_channelpool)
        scale = F.sigmoid(x_out)
        return scale


class CSABlock(nn.Module):
    """
    combine the channel attention and spatial attention to
    build the channel and spatial attention block (CSABlock)
    """

    def __init__(self, in_channel):
        super(CSABlock, self).__init__()
        self.channelattention = ChannelAttention(in_channel)
        self.spatialattention = SpatialAttention()

    def forward(self, x, y):
        """
        x: low-level feature
        y: high-level feature
        """
        h, w = x.size(2), x.size(3)
        channel_scale = self.channelattention(y)
        x_out = x * channel_scale
        spatial_scale = self.spatialattention(x_out)
        x_out = x_out * spatial_scale
        y_up = F.interpolate(y, (h, w), mode="bilinear",
                             align_corners=True)
        out = x_out + y_up
        return out

