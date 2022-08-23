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


class Unet(nn.Module):
    def __init__(self, in_channels=[32, 48, 64, 80, 96], out_channels=[48, 64, 80, 96, 112]):
        super(Unet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # First Unet
        self.conv1_down = nn.Sequential(
            conv2d_bn(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=3, stride=1, padding=1),
        )
        # H x W -> H/2 x W/2
        self.conv2_down = nn.Sequential(
            conv2d_bn(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=3, stride=1, padding=1),
        )
        # H/2 x W/2 -> H/4 x W/4
        self.conv3_down = nn.Sequential(
            conv2d_bn(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=3, stride=1, padding=1),
        )
        # H/4 x W/4 -> H/8 x W/8
        self.conv4_down = nn.Sequential(
            conv2d_bn(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
        )

        self.conv1_up = nn.Sequential(
            conv2d_bn(in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=3, stride=1, padding=1),
        )
        self.deconv1 = deconv2d_bn(in_channels=out_channels[4], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        # H/8 x W/8 -> H/4 x W/4
        self.conv2_up = nn.Sequential(
            conv2d_bn(in_channels=out_channels[3]*2, out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
        )
        self.deconv2 = deconv2d_bn(in_channels=out_channels[3], out_channels=out_channels[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        # H/4 x W/4 -> H/2 x W/2
        self.conv3_up = nn.Sequential(
            conv2d_bn(in_channels=out_channels[2]*2, out_channels=out_channels[2], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=3, stride=1, padding=1),
        )
        self.deconv3 = deconv2d_bn(in_channels=out_channels[2], out_channels=out_channels[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        # H/2 x W/2 -> 32 x H x W
        self.conv4_up = nn.Sequential(
            conv2d_bn(in_channels=out_channels[1]*2, out_channels=out_channels[1], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=3, stride=1, padding=1),
        )
        self.deconv4 = deconv2d_bn(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.post_conv = nn.Sequential(
            conv2d_bn(in_channels=out_channels[0]*2, out_channels=out_channels[0], kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels=out_channels[0], out_channels=in_channels[0], kernel_size=3, stride=1, padding=1),
        )

        
    def forward(self, x, pre_conv=[None, None, None, None]):
        conv1_down = self.conv1_down(x)
        if pre_conv[0] is None:
            conv2_down = self.conv2_down(self.maxpool(conv1_down))  # H x W -> H/2 x W/2
            conv3_down = self.conv3_down(self.maxpool(conv2_down))  # H/2 x W/2 -> H/4 x W/4
            conv4_down = self.conv4_down(self.maxpool(conv3_down))  # H/4 x W/4 -> H/8 x W/8
            conv1_up = self.conv1_up(self.maxpool(conv4_down))  # H/8 x W/8 -> H/16 x W/16
        else:
            conv2_down = self.conv2_down(torch.cat([pre_conv[0], self.maxpool(conv1_down)], dim=1))  # H x W -> H/2 x W/2
            conv3_down = self.conv3_down(torch.cat([pre_conv[1], self.maxpool(conv2_down)], dim=1))  # H/2 x W/2 -> H/4 x W/4
            conv4_down = self.conv4_down(torch.cat([pre_conv[2], self.maxpool(conv3_down)], dim=1))  # H/4 x W/4 -> H/8 x W/8
            conv1_up = self.conv1_up(torch.cat([pre_conv[3], self.maxpool(conv4_down)], dim=1))  # H/8 x W/8 -> H/16 x W/16
        # print(conv4_down.shape)
        # print(self.deconv1(conv1_up).shape)
        conv2_up = self.conv2_up(torch.cat([conv4_down, self.deconv1(conv1_up)], dim=1))  # H/16 x W/16 -> H/8 x W/8
        conv3_up = self.conv3_up(torch.cat([conv3_down, self.deconv2(conv2_up)], dim=1))  # H/8 x W/8 -> H/4 x W/4
        conv4_up = self.conv4_up(torch.cat([conv2_down, self.deconv3(conv3_up)], dim=1))  # H/4 x W/4 -> H/2 x W/2
        post_conv = self.post_conv(torch.cat([conv1_down, self.deconv4(conv4_up)], dim=1))  # H/2 x W/2 -> H x W
        return post_conv, conv4_up, conv3_up, conv2_up, conv1_up


class cascaded_Unet(nn.Module):
    def __init__(self):
        super(cascaded_Unet, self).__init__()
        # H x W -> H/3 x W/3
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=3, padding=1, bias=False)
        self.unet1 = Unet(in_channels=[32, 32, 48, 64, 80], out_channels=[32, 48, 64, 80, 96])
        self.unet2 = Unet(in_channels=[32, 32+48, 48+64, 64+80, 80+96], out_channels=[32, 48, 64, 80, 96])

    def forward(self, x):
        x = self.initial_conv(x)
        feat1, conv4_up, conv3_up, conv2_up, conv1_up = self.unet1(x)
        feat2, _, _, _, _ = self.unet2(x, pre_conv=[conv4_up, conv3_up, conv2_up, conv1_up])
        return feat1, feat2