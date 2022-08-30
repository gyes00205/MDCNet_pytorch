import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
import time
import argparse
from torchsummaryX import summary
from models.dual_encoder import DCU
from models.backbone import cascaded_Unet
from models.cost_aggregation import CostAggregation_2d, CostAggregation_3d

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
