
from torchvision.models import resnet50

from pytorch_pretrained_vit import ViT

import torch.nn as nn
import torch

class Res50(nn.Module):

    def __init__(self, pretrained=True):
        super(Res50, self).__init__()
        self.networks = resnet50(pretrained=pretrained)

        self.conv1 = self.networks.conv1
        self.bn1 = self.networks.bn1
        self.relu = self.networks.relu
        self.maxpool = self.networks.maxpool
        self.layer1 = self.networks.layer1
        self.layer2 = self.networks.layer2
        self.layer3 = self.networks.layer3
        self.layer4 = self.networks.layer4

        self.in_channels = [512, 1024, 2048]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return (x2, x3, x4)

class ViTbackbone(nn.Module):

    def __init__(self, name, pretrained, input_size=1024, input_size_to_transformer=512):
        super(ViTbackbone, self).__init__()
        self.networks = ViT(name=name, pretrained=pretrained)
        self.input_size = input_size
        self.input_size_to_transformer = input_size_to_transformer
        assert int(input_size / input_size_to_transformer) == input_size / input_size_to_transformer
        self.conv_stride = int(input_size / input_size_to_transformer)
        self.input_conv = nn.Conv2d(3, 3, 3, self.conv_stride, 1)
        self.patch_embedding = self.networks.patch_embedding
        self.patch_size = self.patch_embedding.stride[0]
        self.in_channels = (self.patch_embedding.out_channels,)
        if hasattr(self.networks, 'class_token'):
            self.class_token = self.networks.class_token
        if hasattr(self.networks, 'positional_embedding'):
            self.positional_embedding = self.networks.positional_embedding
            self.pretrained_seq_len = self.positional_embedding.pos_embedding.shape[1] - 1
            self.reBuildPositionalEmbedding(int((self.input_size_to_transformer / self.patch_size)**2))

        # self.input_size = int(self.input_size / self.input_conv.stride[0])
        self.layers = self.networks.transformer.blocks

        self.output_conv = nn.ConvTranspose2d(self.in_channels[0], self.in_channels[0], 3, self.conv_stride, 1, 1)

        if hasattr(self.networks, 'pre_logits'):
            self.pre_logits = self.networks.pre_logits


    def reBuildPositionalEmbedding(self, seq_len):
        self.seq_len = seq_len
        if self.pretrained_seq_len < self.seq_len:
            repeat_time = int(self.seq_len / self.pretrained_seq_len)
            self.positional_embedding.pos_embedding = nn.Parameter(torch.cat([
                self.positional_embedding.pos_embedding[:, :1, :],
                self.positional_embedding.pos_embedding[:, 1:, :].repeat_interleave(repeat_time, dim=1)
            ], dim=1))
            repeated_len = self.positional_embedding.pos_embedding.shape[1] - 1
            if repeated_len < self.seq_len:
                plen_l = int((self.seq_len - repeated_len)/2)
                plen_r = self.seq_len - repeated_len - plen_l
                self.positional_embedding.pos_embedding = nn.Parameter(torch.cat([
                    self.positional_embedding.pos_embedding[:, :1, :],
                    torch.nn.functional.pad(
                        self.positional_embedding.pos_embedding[:, 1:, :], (0, 0, plen_l, plen_r)
                    )
                ], dim=1))
        elif self.pretrained_seq_len > self.seq_len:
            self.positional_embedding.pos_embedding = nn.Parameter(self.positional_embedding.pos_embedding[:, :self.seq_len+1, :])


    def forward(self, x):
        _, _, ih, iw = x.shape  # N x B x H x W
        assert ih == iw
        self.downsample_ratio = ih/self.input_size_to_transformer
        x = self.input_conv(x)  # N x B x H/self.conv_stride x W/self.conv_stride
        x = nn.functional.interpolate(x, size=self.input_size_to_transformer)  # N x B x input_size_to_transformer x input_size_to_transformer
        x = self.patch_embedding(x)  # b,d,gh,gw
        b, c, fh, fw = x.shape
        assert fh == fw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1,d

        for m in self.layers:
            x = m(x, None)

        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)

        x = x[:, 1:, :].transpose(2, 1).unflatten(2, (fh, fw))  # N x B x input_size_to_transformer/16 x input_size_to_transformer/16
        # x = x.transpose(2, 1).unflatten(2, (fh, fw))
        # x = self.postprocess(x)
        x = self.output_conv(x)  # N x B x input_size/16 x input_size/16
        if self.downsample_ratio != 0:
            x = nn.functional.interpolate(x, size=int(fh*self.downsample_ratio))  # N x B x H/16 x W/16
        return (x,)