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

class stackhourglass(nn.Module):
    def __init__(self, args):
        super(stackhourglass, self).__init__()
        self.maxdisp = args.maxdisp
        self.num_groups = 20
        self.feature_extraction = cascaded_Unet()
        self.dilated_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.dilated_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.generate_feat_cat = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.cost_agg_3d = CostAggregation_3d(in_channels=32, base_channels=16)
    
    def forward(self, ref, target):
        B, C, H, W = ref.shape
        ref_gwc_feat, target_gwc_feat = dict(), dict()
        _, ref_feat2 = self.feature_extraction(ref)
        _, target_feat2 = self.feature_extraction(target)

        # Second Unet feature generate correlation feature
        ref_l2 = self.dilated_conv1(ref_feat2)
        ref_l3 = self.dilated_conv2(ref_l2)
        target_l2 = self.dilated_conv1(target_feat2)
        target_l3 = self.dilated_conv2(target_l2)
        ref_feat_corr = torch.cat((ref_feat2, ref_l2, ref_l3), dim=1)
        target_feat_corr = torch.cat((target_feat2, target_l2, target_l3), dim=1)

        # Use correlation feature to generate concat feature
        ref_feat_cat = self.generate_feat_cat(ref_feat_corr)
        target_feat_cat = self.generate_feat_cat(target_feat_corr)
        
        # GWC block is used to build 4d cost volume
        ref_gwc_feat['gwc_feature'], ref_gwc_feat['concat_feature'] = ref_feat_corr, ref_feat_cat
        target_gwc_feat['gwc_feature'], target_gwc_feat['concat_feature'] = target_feat_corr, target_feat_cat

        gwc_volume = build_gwc_volume(ref_gwc_feat["gwc_feature"], target_gwc_feat["gwc_feature"], self.maxdisp // 3,
                                      self.num_groups)

        concat_volume = build_concat_volume(ref_gwc_feat["concat_feature"], target_gwc_feat["concat_feature"],
                                            self.maxdisp // 3)
        cost_volume_4d = torch.cat((gwc_volume, concat_volume), dim=1)

        interval_volume = torch.ones(B, 1, H, W, device='cuda')
        disp_range_samples = torch.arange(0, self.maxdisp, requires_grad=False, device='cuda').reshape(1, -1, 1, 1) * interval_volume

        fine_disp = self.cost_agg_3d(
            cost_volume_4d, FineD=self.maxdisp,
            FineH=ref.shape[2],
            FineW=ref.shape[3],
            disp_range_samples=disp_range_samples
        )
        return fine_disp
