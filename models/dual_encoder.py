import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *


class encoder(nn.Module):
    def __init__(self, in_channels):
        super(encoder, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv1 = self.conv1(self.initial_conv(x))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4, conv3, conv2, conv1


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=80, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x=[None, None, None, None], y=[None, None, None, None]):
        deconv1 = self.deconv1(x[0]+y[0])
        deconv2 = self.deconv2(deconv1+x[1]+y[1])
        deconv3 = self.deconv3(deconv2+x[2]+y[2])
        deconv4 = self.deconv4(deconv3+x[3]+y[3])
        return deconv4


class DCU(nn.Module):
    def __init__(self):
        super(DCU, self).__init__()
        self.encoder1 = encoder(in_channels=1)
        self.encoder2 = encoder(in_channels=3)
        self.decoder = decoder()

    def forward(self, rough_disp, error_map):
        conv_disp = self.encoder1(rough_disp)
        conv_err = self.encoder2(error_map)
        disp_range = self.decoder(conv_disp, conv_err)
        return disp_range