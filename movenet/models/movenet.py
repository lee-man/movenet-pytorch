# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# Modified by Min Li
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import cv2

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from .backbone_utils import mobilenet_backbone
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MoveNet(nn.Module):
    '''
    MoveNet from Goolge. Please refer their blog: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html

    '''
    def __init__(self, backbone, heads, head_conv, ft_size=48.0):
        super(MoveNet, self).__init__()
        self.out_channels = 24
        self.backbone = backbone
        self.heads = heads
        self.ft_size = ft_size
        self.weight_to_center = self._generate_center_dist(self.ft_size).unsqueeze(2)
 
        self.dist_y, self.dist_x = self._generate_dist_map(self.ft_size)
        # self.index_17 = torch.arange(0, 17).float()

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, groups=self.out_channels, bias=True),
                  nn.Conv2d(self.out_channels, head_conv, 1, 1, 0, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
            self.__setattr__(head, fc)


    def forward(self, x):
        # specify the device
        device = x.device
        self.weight_to_center = self.weight_to_center.to(device)
        self.dist_y, self.dist_x = self.dist_y.to(device), self.dist_x.to(device)
        
        # conv forward
        x  = x * 0.007843137718737125 - 1.0
        x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        x = ret

        kpt_heatmap, center, kpt_regress, kpt_offset = x['hm_hp'].squeeze(0).permute((1, 2, 0)), x['hm'].squeeze(0).permute((1, 2, 0)), x['hps'].squeeze(0).permute((1, 2, 0)), x['hp_offset'].squeeze(0).permute((1, 2, 0))

        # pose decode
        kpt_heatmap = torch.sigmoid(kpt_heatmap)
        center = torch.sigmoid(center)

        ct_ind = self._top_with_center(center, self.ft_size)
        kpt_coor = self._center_to_kpt(kpt_regress, ct_ind, self.ft_size)

        kpt_top_inds = self._kpt_from_heatmap(kpt_heatmap, kpt_coor, self.ft_size)

        kpt_with_conf = self._kpt_from_offset(kpt_offset, kpt_top_inds, kpt_heatmap, self.ft_size)
        
        return kpt_with_conf

        
    def _draw(self, ft):
        plt.imshow(ft.numpy().reshape(self.ft_size, self.ft_size))
        plt.show()

    def _generate_center_dist(self, ft_size=48, delta=1.8):
        y = torch.arange(ft_size).view(ft_size, 1)
        x = torch.arange(ft_size).view(1, ft_size)
        center_y = ft_size / 2.0
        center_x = ft_size/ 2.0
        y = y - center_y
        x = x - center_x
        weight_to_center = 1 / (torch.sqrt(y * y + x * x) + delta)
        return weight_to_center

    def _generate_dist_map(self, ft_size=48):
        y = torch.arange(ft_size).view(ft_size, 1)
        x = torch.arange(ft_size).view(1, ft_size)
        y = y.repeat((1, ft_size)).unsqueeze(2).float()
        x = x.repeat((ft_size, 1)).unsqueeze(2).float()
        
        return y, x


    def _top_with_center(self, center, ft_size=48):
        scores = center * self.weight_to_center
        top_ind = torch.argmax(scores.view(1, ft_size * ft_size, 1), dim=1)
        return top_ind

    def _center_to_kpt(self, kpt_regress, ct_ind, ft_size=48.0):
        # ct_y = torch.div(ct_ind, ft_size, rounding_mode='floor')
        ct_y = (ct_ind / ft_size).int().float()
        # ct_y = torch.floor_divide(ct_ind.float(), ft_size)
        ct_x = ct_ind - ct_y * ft_size

        kpt_regress = kpt_regress.view(-1, 17, 2)
        ct_ind = ct_ind.unsqueeze(2).expand(ct_ind.size(0), 17, 2)
        kpt_coor = kpt_regress.gather(0, ct_ind).squeeze(0)
        
        kpt_coor = kpt_coor + torch.cat((ct_y, ct_x), dim=1)
        
        return kpt_coor

    def _kpt_from_heatmap(self, kpt_heatmap, kpt_coor, ft_size=48):
        y = self.dist_y - kpt_coor[:, 0].reshape(1, 1, 17)
        x = self.dist_x - kpt_coor[:, 1].reshape(1, 1, 17)
        dist_weight = torch.sqrt(y * y + x * x) + 1.8
        
        scores = kpt_heatmap / dist_weight
        scores = scores.reshape((1, ft_size * ft_size, 17))
        top_inds = torch.argmax(scores, dim=1)
        
        return top_inds
    
    def _kpt_from_offset(self, kpt_offset, kpt_top_inds, kpt_heatmap, size=48.0):
        # kpt_top_inds_float = kpt_top_inds.float()
        # kpts_ys = torch.div(kpt_top_inds, size, rounding_mode='floor')
        kpts_ys = (kpt_top_inds / size).int().float()
        # kpts_ys = torch.floor_divide(kpt_top_inds.float(), size)
        kpts_xs = kpt_top_inds - kpts_ys * size
        kpt_coordinate = torch.stack((kpts_ys.squeeze(0), kpts_xs.squeeze(0)), dim=1)

        kpt_heatmap = kpt_heatmap.view(-1, 17)
        kpt_conf = kpt_heatmap.gather(0, kpt_top_inds).squeeze(0)

        kpt_offset = kpt_offset.view(-1, 17, 2)
        kpt_top_inds = kpt_top_inds.unsqueeze(2).expand(kpt_top_inds.size(0), 17, 2)
        kpt_offset_yx = kpt_offset.gather(0, kpt_top_inds).squeeze(0)

        kpt_coordinate= (kpt_offset_yx + kpt_coordinate) / size
        kpt_with_conf = torch.cat([kpt_coordinate, kpt_conf.unsqueeze(1)], dim=1).reshape((1, 1, 17, 3))

        return kpt_with_conf




def get_pose_net(num_layers, heads, head_conv=96, model_type="movenet_lightning", ft_size=48):
    assert num_layers == 0
    backbone = mobilenet_backbone('mobilenet_v2', pretrained=False, fpn=True, trainable_layers=0, model_type=model_type)
    model = MoveNet(backbone, heads, head_conv=head_conv, ft_size=ft_size)
    return model
