import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
import time
import argparse
from torchsummaryX import summary
from models.dual_encoder import DCU
from models.mdcnet import MDCNet

if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser(description='MDCNet with Flyingthings3d')
    parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
    args = parser.parse_args()
    input1 = torch.rand(1, 3, 240, 528, device=device)
    input2 = torch.rand(1, 3, 240, 528, device=device)
    # summary(MDCNet(args).to(device), input1, input2)
    model = MDCNet(args).to(device)
    start_time = time.time()
    fine_disp, rough_disp = model(input1, input2)
    print(f'time: {time.time() - start_time}')
    print(f'fine disp shape: {fine_disp[0].shape}\nrough disp shape: {rough_disp[0].shape}')