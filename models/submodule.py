import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return conv


def conv3d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )
    return conv


def deconv2d_bn(in_channels, out_channels, kernel_size, stride, padding, output_padding):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return conv


def make_conv3d_block(in_channels, hidden_channels, out_channels, num_layers):
    conv3d_block = [conv3d_bn(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)]
    for _ in range(num_layers - 2):
        conv3d_block += [conv3d_bn(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)]
    conv3d_block += [conv3d_bn(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)]
    return nn.Sequential(*conv3d_block)


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class disparityregression(nn.Module):
    def __init__(self, start, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.arange(start=start, end=maxdisp, device='cuda', requires_grad=False).view(1, -1, 1, 1)

    def forward(self, x):
        out = torch.sum(x * self.disp, dim=1, keepdim=True)
        return out


def disparity_regression_specific(x, disp_values):
    assert len(x.shape) == 4
    return torch.sum(x * disp_values, 1, keepdim=True)


def warp(x, disp):
    '''
    Warp an image tensor right image to left image, according to disparity
    x: [B, C, H, W] right image
    disp: [B, 1, H, W] horizontal shift
    '''
    B, C, H, W = x.shape
    # mesh grid
    '''
    for example: H=4, W=3
    xx =         yy =
    [[0 1 2],    [[0 0 0],    
        [0 1 2],     [1 1 1],
        [0 1 2],     [2 2 2],
        [0 1 2]]     [3 3 3]]
    '''
    xx = torch.arange(0, W, device='cuda').view(1,-1).repeat(H, 1)  # [H, W]
    yy = torch.arange(0, H, device='cuda').view(-1,1).repeat(1, W)  # [H, W]
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
    vgrid = torch.cat((xx, yy), dim=1).float()   # [B, 2, H, W]

    # the correspondence between left and right is that left (i, j) = right (i-d, j)
    vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
    # scale to [-1, 1]
    vgrid[:, 0, :, :] = vgrid[:, 0, :, :] * 2.0 / (W-1) - 1.0
    vgrid[:, 1, :, :] = vgrid[:, 1, :, :] * 2.0 / (H-1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output
