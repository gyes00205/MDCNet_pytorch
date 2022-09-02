import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
import time
import argparse
from torchsummaryX import summary
from models.dual_encoder import DCU
from models.backbone import cascaded_Unet


class CostAggregation_2d(nn.Module):
    def __init__(self, maxdisp):
        super(CostAggregation_2d, self).__init__()
        self.maxdisp = maxdisp
        self.conv1_down = nn.Sequential(
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_down = nn.Sequential(
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1),
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
        )
        self.conv3_down = nn.Sequential(
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1),
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
        )
        self.conv4_down = nn.Sequential(
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1),
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
        )
        self.conv1_up = nn.Sequential(
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1),
            conv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
            deconv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.conv2_up = nn.Sequential(
            conv2d_bn(in_channels=maxdisp*2, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
            deconv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.conv3_up = nn.Sequential(
            conv2d_bn(in_channels=maxdisp*2, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
        )
        self.deconv1 = deconv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4_up = nn.Sequential(
            conv2d_bn(in_channels=maxdisp*2, out_channels=maxdisp, kernel_size=3, stride=1, padding=1),
        )
        self.deconv2 = deconv2d_bn(in_channels=maxdisp, out_channels=maxdisp, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.post_conv = conv2d_bn(in_channels=maxdisp*2, out_channels=maxdisp, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, D, H, W = x.shape
        # print(x.shape)
        conv1_down = self.conv1_down(x)
        conv2_down = self.conv2_down(conv1_down)
        conv3_down = self.conv3_down(conv2_down)
        conv4_down = self.conv4_down(conv3_down)
        conv1_up = self.conv1_up(conv4_down)
        conv2_up = self.conv2_up(torch.cat([conv4_down, conv1_up], dim=1))
        conv3_up = self.conv3_up(torch.cat([conv3_down, conv2_up], dim=1))
        conv4_up = self.conv4_up(torch.cat([conv2_down, self.deconv1(conv3_up)], dim=1))
        post_conv = self.post_conv(torch.cat([conv1_down, self.deconv2(conv4_up)], dim=1))
        # Upsample to original size
        conv3_up = F.interpolate(torch.unsqueeze(conv3_up, dim=1), size=(D*3, H*3, W*3), mode='trilinear', align_corners=True)
        conv4_up = F.interpolate(torch.unsqueeze(conv4_up, dim=1), size=(D*3, H*3, W*3), mode='trilinear', align_corners=True)
        post_conv = F.interpolate(torch.unsqueeze(post_conv, dim=1), size=(D*3, H*3, W*3), mode='trilinear', align_corners=True)

        conv3_up = conv3_up.squeeze(1)  # [B, H/12, W/12]->[B, H, W] low resolution
        conv4_up = conv4_up.squeeze(1)  # [B, H/6, W/6]->[B, H, W] mid resolution
        post_conv = post_conv.squeeze(1)  # [B, H/3, W/3]->[B, H, W] high resolution
        # Disparity regression
        pred1 = F.softmax(conv3_up, dim=1)
        pred1 = disparityregression(start=0, maxdisp=D*3)(pred1)

        pred2 = F.softmax(conv4_up, dim=1)
        pred2 = disparityregression(start=0, maxdisp=D*3)(pred2)

        pred3 = F.softmax(post_conv, dim=1)
        pred3 = disparityregression(start=0, maxdisp=D*3)(pred3)
        
        return pred1, pred2, pred3


class Unet(nn.Module):
    def __init__(self, args):
        super(Unet, self).__init__()
        self.maxdisp = args.maxdisp
        self.feature_extraction = cascaded_Unet()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.rough_disparity_generation = CostAggregation_2d(self.maxdisp//3)
    
    def forward(self, ref, target):
        B, C, H, W = ref.shape
        ref_feat1, _ = self.feature_extraction(ref)
        target_feat1, _ = self.feature_extraction(target)

        # Correlation layer
        ref_feat1, target_feat1 = self.pre_conv(ref_feat1), self.pre_conv(target_feat1)
        cost_volume_3d = build_gwc_volume(ref_feat1, target_feat1, maxdisp=self.maxdisp//3, num_groups=1)
        cost_volume_3d = cost_volume_3d.squeeze(1)  # [B, D/3, H/3, W/3]
        # 2D disparity map generation
        # rough disparity [low(H/12 x H/12), mid(H/6 x H/6), high(H/3 x H/3)]
        # but all upsample to H x W
        rough_disp = self.rough_disparity_generation(cost_volume_3d)  # [low resolution, mid resolution, high resolution]
        return rough_disp
