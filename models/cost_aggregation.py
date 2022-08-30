import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *


def pad(x, y):
    diffY = x.shape[2] - y.shape[2]
    diffX = x.shape[3] - y.shape[3]
    y_new = F.pad(y, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    return y_new


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = conv3d_bn(inplanes, inplanes*2, kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(inplanes*2)
        )

        self.conv3 = conv3d_bn(inplanes*2, inplanes*2, kernel_size=3, stride=2, padding=1)

        self.conv4 = conv3d_bn(inplanes*2, inplanes*2, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post


class CostAggregation_3d(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(CostAggregation_3d, self).__init__()

        self.dres0 = nn.Sequential(
            conv3d_bn(in_channels, base_channels, 3, 1, 1),
            conv3d_bn(base_channels, base_channels, 3, 1, 1),
        )

        self.dres1 = nn.Sequential(
            conv3d_bn(base_channels, base_channels, 3, 1, 1),
            nn.Conv3d(base_channels, base_channels, 3, 1, 1),
            nn.BatchNorm3d(base_channels)
        )

        self.dres2 = hourglass(base_channels)

        self.dres3 = hourglass(base_channels)

        self.dres4 = hourglass(base_channels)

        self.classif0 = nn.Sequential(
            conv3d_bn(base_channels, base_channels, 3, 1, 1),
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif1 = nn.Sequential(
            conv3d_bn(base_channels, base_channels, 3, 1, 1),
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif2 = nn.Sequential(
            conv3d_bn(base_channels, base_channels, 3, 1, 1),
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif3 = nn.Sequential(
            conv3d_bn(base_channels, base_channels, 3, 1, 1),
            nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

    def forward(self, cost, FineD, FineH, FineW, disp_range_samples):

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre2, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        
        cost1 = F.interpolate(cost1, [FineD, FineH, FineW], mode='trilinear', align_corners=True)
        cost2 = F.interpolate(cost2, [FineD, FineH, FineW], mode='trilinear', align_corners=True)
        cost3 = F.interpolate(cost3, [FineD, FineH, FineW], mode='trilinear', align_corners=True)

        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparity_regression_specific(pred1, disp_range_samples)

        cost2 = torch.squeeze(cost2,1)
        pred2 = F.softmax(cost2,dim=1)
        pred2 = disparity_regression_specific(pred2, disp_range_samples)

        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparity_regression_specific(pred3, disp_range_samples)

        return pred1, pred2, pred3


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