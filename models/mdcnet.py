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


class GetCostVolume(nn.Module):
    def __init__(self):
        super(GetCostVolume, self).__init__()

    def get_warped_feats(self, x, y, disp_range_samples, ndisp):
        bs, channels, height, width = y.size()

        mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
        mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
        mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

        cur_disp_coords_y = mh
        cur_disp_coords_x = mw - disp_range_samples

        # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
        # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

        coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
        coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
        grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

        y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)


        # a littel difference, no zeros filling
        x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)
        x_warped = x_warped.transpose(0, 1) #(C, B, D, H, W)
        #x1 = x2 + d >= d
        x_warped[:, mw < disp_range_samples] = 0
        x_warped = x_warped.transpose(0, 1) #(B, C, D, H, W)

        return x_warped, y_warped

    def build_concat_volume(self, x, y, disp_range_samples, ndisp):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        concat_cost = x.new().resize_(bs, channels * 2, ndisp, height, width).zero_()  # (B, 2C, D, H, W)

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp)
        concat_cost[:, x.size()[1]:, :, :, :] = y_warped
        concat_cost[:, :x.size()[1], :, :, :] = x_warped

        return concat_cost

    def build_gwc_volume(self, x, y, disp_range_samples, ndisp, num_groups):
        assert (x.is_contiguous() == True)
        bs, channels, height, width = x.size()

        x_warped, y_warped = self.get_warped_feats(x, y, disp_range_samples, ndisp) #(B, C, D, H, W)

        assert channels % num_groups == 0
        channels_per_group = channels // num_groups
        gwc_cost = (x_warped * y_warped).view([bs, num_groups, channels_per_group, ndisp, height, width]).mean(dim=2)  #(B, G, D, H, W)

        return gwc_cost

    def forward(self, features_left, features_right, disp_range_samples, ndisp, num_groups):
        # bs, channels, height, width = features_left["gwc_feature"].size()
        gwc_volume = self.build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"],
                                           disp_range_samples, ndisp, num_groups)

        concat_volume = self.build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                 disp_range_samples, ndisp)

        volume = torch.cat((gwc_volume, concat_volume), 1)   #(B, C+G, D, H, W)

        return volume
        

class MDCNet(nn.Module):
    def __init__(self, args):
        super(MDCNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.feature_extraction = cascaded_Unet()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.rough_disparity_generation = CostAggregation_2d(self.maxdisp//3)
        self.dual_encoder = DCU()
        self.dilated_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.dilated_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.generate_feat_cat = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.get_cv = GetCostVolume()
        self.cost_agg_3d = CostAggregation_3d(in_channels=32, base_channels=16)

    def forward(self, ref, target):
        B, C, H, W = ref.shape
        ref_gwc_feat, target_gwc_feat = dict(), dict()
        ref_feat1, ref_feat2 = self.feature_extraction(ref)
        target_feat1, target_feat2 = self.feature_extraction(target)
        # print(f'ref feat1 shape: {ref_feat1.shape}')
        # print(f'ref feat2 shape: {ref_feat2.shape}')

        # Correlation layer
        ref_feat1, target_feat1 = self.pre_conv(ref_feat1), self.pre_conv(target_feat1)
        cost_volume_3d = build_gwc_volume(ref_feat1, target_feat1, maxdisp=self.maxdisp//3, num_groups=1)
        cost_volume_3d = cost_volume_3d.squeeze(1)  # [B, D/3, H/3, W/3]
        # print(f'cost volume shape: {cost_volume_3d.shape}')

        # 2D disparity map generation
        # rough disparity [low(H/12 x H/12), mid(H/6 x H/6), high(H/3 x H/3)]
        # but all upsample to H x W
        rough_disp = self.rough_disparity_generation(cost_volume_3d)  # [low resolution, mid resolution, high resolution]
        recon_ref = warp(target, rough_disp[-1])
        error_map = recon_ref - ref
        disp_range = self.dual_encoder(rough_disp[-1], error_map)  # [B, 24, H, W]
        disp_range = F.interpolate(torch.unsqueeze(disp_range, dim=1), size=(8, H//3, W//3), mode='trilinear', align_corners=True) / 3.0 # [B, 1, 8, H/3, W/3]
        disp_range = disp_range.squeeze(1)  # [B, 8, H/3, W/3]

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

        cost_volume_4d = self.get_cv(
            ref_gwc_feat, target_gwc_feat,
            disp_range_samples=disp_range,
            ndisp=8, num_groups=20
        )
        disp_range_upscale = F.interpolate(
            (disp_range*3).unsqueeze(1),
            size=[24, ref.shape[2], ref.shape[3]],
            mode='trilinear',
            align_corners=True
        ).squeeze(1)
        fine_disp = self.cost_agg_3d(
            cost_volume_4d, FineD=24,
            FineH=ref.shape[2],
            FineW=ref.shape[3],
            disp_range_samples=disp_range_upscale
        )
        # [low resolution, mid resolution, high resolution]

        return rough_disp, fine_disp
